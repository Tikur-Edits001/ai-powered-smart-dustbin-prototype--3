// api/analyze-waste.js
//
// Serverless endpoint for the EcoSort AI Waste Scanner.
// Deploy this on Vercel (or adapt for Netlify/AWS Lambda — see README.md)
// alongside index.html. It receives an uploaded image from the browser,
// forwards it to a vision-capable AI model (Anthropic's Claude), and
// returns strict, validated JSON describing the waste item.
//
// The API key NEVER touches the browser — it only lives in this
// server-side function, read from the ANTHROPIC_API_KEY environment
// variable you configure in your hosting provider's dashboard.

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const MODEL = 'claude-sonnet-4-6';
const MAX_IMAGE_BYTES = 8 * 1024 * 1024; // 8MB, matches the frontend cap
const UPSTREAM_TIMEOUT_MS = 25000;

const SYSTEM_PROMPT = `You are a waste-classification vision system for a recycling app.
Look carefully at the image and identify the main waste item(s) visible.

Respond with ONLY a single valid JSON object, no prose, no markdown fences, matching this exact shape:

{
  "is_waste": boolean,
  "detected_object": string,
  "waste_category": one of ["Plastic","Paper","Glass","Metal","Organic Waste","E-Waste","Hazardous Waste","General/Residual Waste"],
  "recommended_bin": string,
  "confidence": number between 0 and 100,
  "environmental_impact": string (1-2 sentences),
  "probabilities": { "<category>": number, ... }  // percentages across the categories above, roughly summing to 100
}

Rules:
- If the image clearly does NOT show waste/trash of any kind (e.g. a person, pet, landscape, unrelated object), set "is_waste" to false, "detected_object" to "" and "confidence" to 0. Still return valid JSON in the same shape.
- If the image is too blurry, dark, obstructed, or otherwise impossible to classify with reasonable certainty, set "confidence" below 40, and still fill the other fields with your best guess.
- Battery and e-waste items belong to "E-Waste". Textile/fabric and anything with no clear recyclable stream belongs to "General/Residual Waste".
- Use your best judgement for the single most prominent waste item if multiple items are visible.
- Never include any text outside the JSON object.`;

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    // Matches the frontend's expected "not connected" messaging.
    return res.status(503).json({
      error: 'AI service is not connected. Configure the vision API endpoint to enable real image analysis.'
    });
  }

  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch (_) { body = null; }
  }

  const imageDataUrl = body && body.image;
  if (!imageDataUrl || typeof imageDataUrl !== 'string' || !imageDataUrl.startsWith('data:image/')) {
    return res.status(400).json({ error: 'No valid image was provided. Please upload an image and try again.' });
  }

  const match = imageDataUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)$/);
  if (!match) {
    return res.status(400).json({ error: 'Unsupported image format. Please upload a PNG or JPG image.' });
  }
  const mediaType = match[1];
  const base64Data = match[2];

  const approxBytes = Math.round(base64Data.length * 0.75);
  if (approxBytes > MAX_IMAGE_BYTES) {
    return res.status(413).json({ error: 'The image is too large to analyze. Please use an image under 8MB.' });
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const upstream = await fetch(ANTHROPIC_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 500,
        system: SYSTEM_PROMPT,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: mediaType, data: base64Data } },
              { type: 'text', text: 'Classify the waste item(s) in this image. Respond with only the JSON object described in your instructions.' }
            ]
          }
        ]
      }),
      signal: controller.signal
    });

    if (!upstream.ok) {
      const errText = await upstream.text().catch(() => '');
      console.error('Upstream vision API error:', upstream.status, errText);
      if (upstream.status === 401 || upstream.status === 403) {
        return res.status(503).json({ error: 'AI service is not connected. Configure the vision API endpoint to enable real image analysis.' });
      }
      if (upstream.status === 429) {
        return res.status(429).json({ error: 'The AI service is busy right now. Please try again in a moment.' });
      }
      return res.status(502).json({ error: 'The AI vision service returned an error. Please try again.' });
    }

    const payload = await upstream.json();
    const textBlock = Array.isArray(payload.content)
      ? payload.content.find((b) => b.type === 'text')
      : null;
    const rawText = textBlock && textBlock.text ? textBlock.text.trim() : '';

    let parsed;
    try {
      // Strip accidental markdown fences just in case the model adds them.
      const cleaned = rawText.replace(/^```json\s*|^```\s*|```$/gim, '').trim();
      parsed = JSON.parse(cleaned);
    } catch (parseErr) {
      console.error('Failed to parse model JSON:', rawText);
      return res.status(502).json({ error: 'The AI service returned an invalid response. Please try again.' });
    }

    const result = validateModelResult(parsed);
    return res.status(200).json(result);
  } catch (err) {
    if (err.name === 'AbortError') {
      return res.status(504).json({ error: 'The analysis request timed out. Please try again.' });
    }
    console.error('analyze-waste handler error:', err);
    return res.status(500).json({ error: 'Something went wrong while analyzing the image. Please try again.' });
  } finally {
    clearTimeout(timeoutId);
  }
};

const VALID_CATEGORIES = [
  'Plastic', 'Paper', 'Glass', 'Metal',
  'Organic Waste', 'E-Waste', 'Hazardous Waste', 'General/Residual Waste'
];

// Server-side validation mirrors the frontend's checks so malformed or
// out-of-range model output never reaches the browser as if it were trustworthy.
function validateModelResult(parsed) {
  const isWaste = parsed.is_waste !== false;

  let confidence = Number(parsed.confidence);
  if (!Number.isFinite(confidence)) confidence = 0;
  confidence = Math.max(0, Math.min(100, confidence));

  const category = VALID_CATEGORIES.includes(parsed.waste_category)
    ? parsed.waste_category
    : 'General/Residual Waste';

  let probabilities = {};
  if (parsed.probabilities && typeof parsed.probabilities === 'object') {
    for (const [key, val] of Object.entries(parsed.probabilities)) {
      const num = Number(val);
      if (VALID_CATEGORIES.includes(key) && Number.isFinite(num)) {
        probabilities[key] = Math.max(0, Math.min(100, num));
      }
    }
  }
  if (Object.keys(probabilities).length === 0) {
    probabilities[category] = confidence;
  }

  return {
    is_waste: isWaste,
    detected_object: typeof parsed.detected_object === 'string' ? parsed.detected_object.trim() : '',
    waste_category: category,
    recommended_bin: typeof parsed.recommended_bin === 'string' ? parsed.recommended_bin.trim() : '',
    confidence,
    environmental_impact: typeof parsed.environmental_impact === 'string' ? parsed.environmental_impact.trim() : '',
    probabilities
  };
}
