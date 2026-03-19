const express = require('express');
const axios = require('axios');
const { getServiceUrl } = require('../config/runtime');

const router = express.Router();
const THERAPY_SERVICE_URL = getServiceUrl('THERAPY_URL', 5002);

const PROXY_PREFIXES = [
  '/therapy/health',
  '/fluency-exercises',
  '/language-exercises',
  '/receptive-exercises',
  '/articulation-exercises'
];

function shouldProxy(path) {
  return PROXY_PREFIXES.some(prefix => path === prefix || path.startsWith(`${prefix}/`));
}

function buildForwardHeaders(headers) {
  const allowedHeaders = ['authorization', 'content-type'];
  const forwarded = {};

  allowedHeaders.forEach(name => {
    if (headers[name]) {
      forwarded[name] = headers[name];
    }
  });

  return forwarded;
}

router.use(async (req, res, next) => {
  if (!shouldProxy(req.path)) {
    return next();
  }

  try {
    const targetUrl = `${THERAPY_SERVICE_URL}${req.originalUrl}`;
    const response = await axios({
      method: req.method,
      url: targetUrl,
      headers: buildForwardHeaders(req.headers),
      data: ['GET', 'HEAD'].includes(req.method) ? undefined : req.body,
      timeout: 30000,
      validateStatus: () => true
    });

    return res.status(response.status).json(response.data);
  } catch (error) {
    return res.status(503).json({
      success: false,
      message: 'Therapy service proxy request failed',
      error: error.message
    });
  }
});

module.exports = router;
