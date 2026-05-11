import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import requests
import urllib3

from . import config


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class LlmClient:
    def __init__(
        self,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        max_retries: int = None,
        retry_delay: float = None,
        backoff_multiplier: float = None,
        timeout: Union[Tuple[int, int], int, None] = None,
        temperature: float = None,
        max_tokens: Optional[int] = None,
        verify_ssl: bool = None,
        extra_body: Optional[Dict] = None,
    ):
        self._url = (api_base if api_base is not None else config.LLM_API_BASE).rstrip('/')
        self._model = model if model is not None else config.LLM_MODEL
        self._api_key = api_key if api_key is not None else config.LLM_API_KEY

        if not self._url or not self._model:
            raise ValueError('LLM api_base/model is required (pass args or set llm_client/config.py constants).')

        self._max_retries = int(max_retries if max_retries is not None else config.LLM_MAX_RETRIES)
        self._retry_delay = float(retry_delay if retry_delay is not None else config.LLM_RETRY_DELAY)
        self._backoff_multiplier = float(
            backoff_multiplier if backoff_multiplier is not None else config.LLM_BACKOFF_MULTIPLIER
        )
        self._timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        self._temperature = float(temperature if temperature is not None else config.LLM_TEMPERATURE)
        self._max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        self._verify_ssl = bool(verify_ssl if verify_ssl is not None else config.LLM_VERIFY_SSL)
        self._extra_body = extra_body if extra_body is not None else (config.LLM_EXTRA_BODY or {})

        if not self._verify_ssl:
            logger.warning('SSL certificate verification is DISABLED. Use only in trusted networks.')

    @staticmethod
    def _clean_response(content: str) -> str:
        if not content:
            return ''
        if '</think>' in content:
            content = content.split('</think>', 1)[1]

        content = content.strip()
        if content.startswith('```'):
            lines = content.split('\n', 1)
            content = lines[1] if len(lines) > 1 else ''
        if content.endswith('```'):
            content = content.rsplit('```', 1)[0]
        return content.strip()

    def chat(self, prompt: str, *, stop: Optional[List[str]] = None, temperature: Optional[float] = None) -> str:
        last_exception: Optional[BaseException] = None
        delay = self._retry_delay

        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._call_llm(prompt, stop=stop, temperature=temperature)
                max_log_chars = 800
                preview = response if len(response) <= max_log_chars else (response[:max_log_chars] + '...<truncated>')
                logger.info('LLM response (len=%d chars, preview_len=%d):\n%s', len(response), len(preview), preview)
                return response
            except Exception as e:  # noqa: BLE001
                last_exception = e
                logger.warning(
                    'LLM call failed (attempt %d/%d): %s',
                    attempt,
                    self._max_retries,
                    e,
                )
                if attempt < self._max_retries:
                    time.sleep(delay)
                    delay *= self._backoff_multiplier
        
        logger.error(f"LLM call failed after {self._max_retries} retries")
        raise RuntimeError(f'LLM call failed after {self._max_retries} retries: {last_exception}')

    def _call_llm(self, prompt: str, *, stop: Optional[List[str]] = None, temperature: Optional[float] = None) -> str:
        headers = {
            'Content-Type': 'application/json',
        }
        if self._api_key:
            headers['Authorization'] = f'Bearer {self._api_key}'

        body: Dict[str, object] = {
            'model': self._model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': float(self._temperature if temperature is None else temperature),
            'stream': bool(getattr(config, 'LLM_STREAM', False)),
        }
        if self._max_tokens is not None:
            body['max_tokens'] = int(self._max_tokens)
        if stop:
            body['stop'] = stop
        if self._extra_body:
            body.update(self._extra_body)

        is_stream = bool(body.get('stream'))
        logger.info(
            'Sending %s request to %s, model=%s, temperature=%s, max_tokens=%s, extra_body=%s',
            'streaming' if is_stream else 'non-streaming',
            self._url,
            self._model,
            body.get('temperature'),
            self._max_tokens,
            self._extra_body,
        )

        resp = requests.post(
            self._url,
            headers=headers,
            data=json.dumps(body, ensure_ascii=False),
            stream=is_stream,
            timeout=self._timeout,
            verify=self._verify_ssl,
        )
        resp.raise_for_status()

        if not is_stream:
            data = resp.json()
            choices = data.get('choices') or []
            if not choices:
                raise RuntimeError(f'LLM response missing choices: {data}')
            msg = choices[0].get('message') or {}
            content = msg.get('content')
            if content is None:
                raise RuntimeError(f'LLM response missing message.content: {data}')
            return self._clean_response(str(content))

        # SSE streaming (OpenAI-compatible)
        content_parts: List[str] = []
        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith('data:'):
                    continue
                data_str = line[len('data:') :].strip()
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get('choices') or []
                if not choices:
                    continue
                delta = (choices[0].get('delta') or {})
                piece = delta.get('content')
                if piece:
                    content_parts.append(str(piece))
        finally:
            resp.close()

        return self._clean_response(''.join(content_parts))
