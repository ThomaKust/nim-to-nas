// server.js — Robust Hybrid OpenAI ↔ NIM Proxy
// Express 5 Compatible
// Fixes: auth bypass, startup DDoS, silent stream failures, memory leaks, Express 5 deprecations

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual, createHmac, randomBytes } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const RP_ENGINE_URL = process.env.RP_ENGINE_URL;
const RP_ENGINE_SECRET = process.env.RP_ENGINE_SECRET;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

// Temporary, privacy-conscious request diagnostics. Enable with DIAGNOSTICS=true.
// The diagnostic HMAC key is random per server start, so message fingerprints cannot
// be used to recover message text and only compare consistently during this run.
const DIAGNOSTICS = process.env.DIAGNOSTICS === 'true';
const DIAGNOSTIC_HASH_KEY = DIAGNOSTICS ? randomBytes(32) : null;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };

  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');

  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}

validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'z-ai/glm-5.3-flash',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'gpt-3.5o': 'moonshotai/kimi-k3',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'glm-5.3': 'z-ai/glm-5.3',
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'm2.7': 'minimaxai/minimax-m2.7',
  'm3': 'minimaxai/minimax-m3',
  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ─── Safe request diagnostics (temporary) ───────────────────────────────────
// Never log actual chat text, API keys, authorization headers, cookies, or raw IDs.
// We log only field names/types and keyed fingerprints for candidate identifiers and
// message contents, so repeated values can be compared during a single server run.
function diagnosticFingerprint(value) {
  let serialized;
  try {
    serialized = typeof value === 'string' ? value : JSON.stringify(value);
  } catch {
    serialized = String(value);
  }
  return createHmac('sha256', DIAGNOSTIC_HASH_KEY)
    .update(serialized ?? 'undefined')
    .digest('hex')
    .slice(0, 10);
}

function diagnosticType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return `array(length=${value.length})`;
  if (typeof value === 'string') return `string(length=${value.length})`;
  if (typeof value === 'object') return 'object';
  return typeof value;
}

function looksLikeIdentifier(name) {
  return /(chat|conversation|thread|session|dialog|turn|message|uuid|request.?id|(?:^|[-_])id$)/i.test(name);
}

function collectDiagnosticFields(value, path = 'body', depth = 0, out = []) {
  if (out.length >= 160 || depth > 7 || value === null || typeof value !== 'object') return out;

  const entries = Array.isArray(value)
    ? value.slice(0, 120).map((item, index) => [String(index), item])
    : Object.entries(value);

  for (const [key, child] of entries) {
    if (out.length >= 160) break;
    const fieldPath = Array.isArray(value) ? `${path}[${key}]` : `${path}.${key}`;
    const lowerKey = String(key).toLowerCase();

    // Messages are summarized separately with role and keyed content fingerprints.
    if (lowerKey === 'messages') {
      out.push(`${fieldPath}: ${diagnosticType(child)} [summarized separately]`);
      continue;
    }

    // Content-like fields are listed by type only. Their text is never logged.
    if (['content', 'text', 'prompt', 'system_prompt', 'instructions', 'description', 'persona', 'authorization', 'cookie', 'api_key', 'apikey', 'token', 'secret'].includes(lowerKey)) {
      out.push(`${fieldPath}: ${diagnosticType(child)} [value omitted]`);
      continue;
    }

    if (looksLikeIdentifier(String(key))) {
      const fingerprint = (typeof child === 'string' || typeof child === 'number')
        ? ` fingerprint=${diagnosticFingerprint(child)}`
        : '';
      out.push(`${fieldPath}: ${diagnosticType(child)}${fingerprint}`);
    } else {
      out.push(`${fieldPath}: ${diagnosticType(child)}`);
    }

    if (child && typeof child === 'object') {
      collectDiagnosticFields(child, fieldPath, depth + 1, out);
    }
  }
  return out;
}

function collectCandidateIdentifiers(value, path = 'message', depth = 0, out = []) {
  if (out.length >= 40 || depth > 5 || value === null || typeof value !== 'object') return out;
  const entries = Array.isArray(value)
    ? value.slice(0, 30).map((item, index) => [String(index), item])
    : Object.entries(value);
  for (const [key, child] of entries) {
    if (out.length >= 40) break;
    const lowerKey = String(key).toLowerCase();
    if (['content', 'text', 'prompt', 'system_prompt', 'instructions', 'description', 'persona', 'authorization', 'cookie', 'api_key', 'apikey', 'token', 'secret'].includes(lowerKey)) continue;
    const fieldPath = Array.isArray(value) ? `${path}[${key}]` : `${path}.${key}`;
    if (looksLikeIdentifier(String(key))) {
      out.push({
        path: fieldPath,
        type: diagnosticType(child),
        fingerprint: (typeof child === 'string' || typeof child === 'number')
          ? diagnosticFingerprint(child)
          : undefined
      });
    }
    if (child && typeof child === 'object') collectCandidateIdentifiers(child, fieldPath, depth + 1, out);
  }
  return out;
}

function logRequestDiagnostics(req) {
  if (!DIAGNOSTICS) return;

  const body = req.body && typeof req.body === 'object' ? req.body : {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const headerIdentifiers = Object.entries(req.headers || {})
    .filter(([name]) => looksLikeIdentifier(name))
    .map(([name, value]) => ({
      name,
      type: diagnosticType(value),
      fingerprint: (typeof value === 'string' || typeof value === 'number')
        ? diagnosticFingerprint(value)
        : undefined
    }));
  const queryIdentifiers = Object.entries(req.query || {})
    .filter(([name]) => looksLikeIdentifier(name))
    .map(([name, value]) => ({
      name,
      type: diagnosticType(value),
      fingerprint: (typeof value === 'string' || typeof value === 'number')
        ? diagnosticFingerprint(value)
        : undefined
    }));

  const messageSummaries = messages.map((message, index) => {
    const role = typeof message?.role === 'string' ? message.role : 'unknown';
    const contentExists = message && Object.prototype.hasOwnProperty.call(message, 'content');
    return {
      index,
      role,
      keys: message && typeof message === 'object' ? Object.keys(message).sort() : [],
      contentType: contentExists ? diagnosticType(message.content) : 'missing',
      // Fingerprint only: the actual content is never written to logs.
      contentFingerprint: contentExists ? diagnosticFingerprint(message.content) : null,
      candidateIdentifiers: collectCandidateIdentifiers(message, `messages[${index}]`)
    };
  });

  const report = {
    marker: 'JANITOR_REQUEST_DIAGNOSTICS',
    time: new Date().toISOString(),
    method: req.method,
    path: req.path,
    bodyTopLevelKeys: Object.keys(body).sort(),
    queryKeys: Object.keys(req.query || {}).sort(),
    headerNames: Object.keys(req.headers || {}).sort(),
    messageCount: messages.length,
    messageRoles: messages.map(message => typeof message?.role === 'string' ? message.role : 'unknown'),
    // Compare the same index across requests to see if messages were edited/removed.
    messageSummaries,
    candidateIdentifierHeaders: headerIdentifiers,
    candidateIdentifierQueryFields: queryIdentifiers,
    bodyFieldPaths: collectDiagnosticFields(body)
  };

  console.log(`[DIAGNOSTICS] ${JSON.stringify(report)}`);
}

// FIX: Extract token AFTER "Bearer " prefix, compare only the token
// Prevents bypass when CLIENT_AUTH_KEY is empty (expected would be "Bearer " which is 7 chars)
function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}
app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);

  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

// FIX: Use lightweight model listing instead of burning inference quota
// If NIM doesn't support /models, skip validation entirely rather than DDoS-ing yourself
async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];

    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── RP Engine: Memory connection test ─────────────────────────────────────

async function testMemoryConnection() {
  if (!RP_ENGINE_URL || !RP_ENGINE_SECRET) {
    console.warn('[MEMORY] RP_ENGINE_URL or RP_ENGINE_SECRET is missing.');
    return;
  }

  try {
    const baseUrl = RP_ENGINE_URL.replace(/\/+$/, '');

    const response = await axios.get(`${baseUrl}/test`, {
      headers: {
        Authorization: `Bearer ${RP_ENGINE_SECRET}`
      },
      timeout: 10000
    });

    if (response.data?.success) {
      console.log(
        '[MEMORY] Worker connection successful. Records:',
        response.data.data?.length ?? 0
      );
    } else {
      console.warn('[MEMORY] Worker returned an unexpected response.');
    }

  } catch (error) {
    console.error(
      '[MEMORY] Connection failed:',
      error.response?.status || error.code || error.message
    );
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

// FIX: Wrap res.write in try/catch to prevent crashes on closed sockets
function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callWithFallback(baseRequest, models) {
  let lastError = null;

  for (const model of models) {
    try {
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );

      return { response: res, model };

    } catch (err) {
      lastError = err;
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.1.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      created: Date.now(),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  logRequestDiagnostics(req);

  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    const primaryModel = MODEL_MAPPING[model] || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
    const modelChain = [primaryModel, ...FALLBACK_MODELS];

    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      extra_body: ENABLE_THINKING_MODE
        ? { chat_template_kwargs: { thinking: true } }
        : undefined
    };

    const { response, model: usedModel } = await callWithFallback(baseRequest, modelChain);
    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        if (line.includes('[DONE]')) {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            let content = delta.content || '';
            const reasoning = delta.reasoning_content;

            if (SHOW_REASONING) {
              if (reasoning && !reasoningOpen) {
                content = `<thinking>\n${reasoning.replace(/\n/g, '\\n')}`;
                reasoningOpen = true;
              } else if (reasoning) {
                content = reasoning.replace(/\n/g, '\\n');
              }

              if (delta.content && reasoningOpen) {
                content += `\n</thinking>\n\n${delta.content}`;
                reasoningOpen = false;
              }
            }

            delta.content = content;
            delete delta.reasoning_content;
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);

        } catch (parseErr) {
          // FIX: Don't silently swallow—send error to client so they know data was lost
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Upstream sent malformed chunk', 
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            } 
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({ 
            error: { 
              message: 'Stream buffer overflow', 
              type: 'stream_error' 
            } 
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();

        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);

        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      // FIX: Check req.destroyed (Node/Express 5) 
      // Don't destroy already-finished streams
      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;

        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }

        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: (response.data.choices || []).map((choice, i) => {
          let content = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            const safeReasoning = choice.message.reasoning_content.replace(/\n/g, '\\n');
            content = `<thinking>\n${safeReasoning}\n</thinking>\n\n${content}`;
          }

          return {
            index: i,
            message: {
              role: choice.message?.role || 'assistant',
              content,
              tool_calls: choice.message?.tool_calls
            },
            finish_reason: choice.finish_reason || 'stop'
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    // Clean up upstream stream if we have it
    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

// FIX: Express 5 named wildcard — but use proper 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);

  // Run validation after server starts, non-blocking
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });

  // Test RP Engine memory connection
  testMemoryConnection();
});
