const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');
const axios = require('axios');
const connectDB = require('./config/database');
const { getAllowedOrigins, getServiceUrl, isProduction, requireJwtSecret } = require('./config/runtime');

// Load env vars
dotenv.config();

requireJwtSecret();

const app = express();
const allowedOrigins = getAllowedOrigins();
const gaitServiceUrl = getServiceUrl('GAIT_ANALYSIS_URL', 5001);
const therapyServiceUrl = getServiceUrl('THERAPY_URL', 5002);

app.disable('x-powered-by');
app.set('trust proxy', 1);

app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-site');
  next();
});

// Body parser middleware
// Gait analysis requests can contain large sensor arrays, so use a
// higher payload limit than the Express default.
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: false, limit: '10mb' }));

// Enable CORS
app.use(cors({
  origin(origin, callback) {
    if (!origin) {
      return callback(null, true);
    }

    if (!isProduction) {
      return callback(null, true);
    }

    if (allowedOrigins.includes(origin)) {
      return callback(null, true);
    }

    return callback(new Error('Origin not allowed by CORS'));
  },
  credentials: true
}));

// Route files (loaded after app is created)
const authRoutes = require('./routes/authRoutes');
const gaitRoutes = require('./routes/gaitRoutes');
const exerciseRoutes = require('./routes/exerciseRoutes');
const adminRoutes = require('./routes/adminRoutes');
const articulationAssessmentRoutes = require('./routes/articulationAssessment');
const articulationRoutes = require('./routes/articulationRoutes');
const fluencyAssessmentRoutes = require('./routes/fluencyAssessment');
const fluencyRoutes = require('./routes/fluencyRoutes');
const receptiveRoutes = require('./routes/receptiveRoutes');
const expressiveRoutes = require('./routes/expressiveRoutes');
const speechRoutes = require('./routes/speechRoutes');
const healthRoutes = require('./routes/healthRoutes');
const successStoryRoutes = require('./routes/successStoryRoutes');
const therapistRoutes = require('./routes/therapistRoutes');
const appointmentRoutes = require('./routes/appointmentRoutes');
const diagnosticComparisonRoutes = require('./routes/diagnosticComparisonRoutes');
const therapyProxyRoutes = require('./routes/therapyProxyRoutes');

app.locals.db = null;

// Mount routers
app.use('/api/auth', authRoutes);
app.use('/api', therapyProxyRoutes);
app.use('/api/gait', gaitRoutes);
app.use('/api/exercises', exerciseRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/health', healthRoutes);  // Health logs and summary
app.use('/api/success-stories', successStoryRoutes);  // Success stories
app.use('/api/articulation', articulationRoutes);  // Progress & exercises
app.use('/api/articulation', articulationAssessmentRoutes);  // Recording assessment
app.use('/api/fluency', fluencyRoutes);  // Fluency progress & exercises
app.use('/api/fluency', fluencyAssessmentRoutes);  // Fluency assessment
app.use('/api/receptive', receptiveRoutes);  // Receptive language therapy
app.use('/api/expressive', expressiveRoutes);  // Expressive language therapy
app.use('/api/speech', speechRoutes);  // Overall speech improvement prediction
app.use('/api/therapist', therapistRoutes);  // Therapist analytics and reports
app.use('/api', appointmentRoutes);  // Appointment scheduling (patient + therapist + shared)
app.use('/api', diagnosticComparisonRoutes);  // Diagnostic comparison (facility vs home)

// Serve uploaded files
app.use('/uploads', express.static('uploads'));

// Admin progress endpoints (separate registration for cleaner URLs)
app.use('/api/articulation-progress', articulationRoutes);
app.use('/api/fluency-progress', fluencyRoutes);
app.use('/api/receptive-progress', receptiveRoutes);
app.use('/api/language-progress', expressiveRoutes);

// Health check route
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'CVACare Backend API is running',
    version: '1.0.0',
    database: app.locals.db ? 'connected' : 'connecting'
  });
});

app.get('/healthz', (req, res) => {
  res.json({
    success: true,
    service: 'node-api',
    status: 'healthy'
  });
});

app.get('/readyz', async (req, res) => {
  const databaseReady = Boolean(app.locals.db);

  try {
    const [gaitHealth, therapyHealth] = await Promise.all([
      axios.get(`${gaitServiceUrl}/health`, { timeout: 5000 }),
      axios.get(`${therapyServiceUrl}/api/therapy/health`, { timeout: 5000 })
    ]);

    if (!databaseReady) {
      return res.status(503).json({
        success: false,
        service: 'node-api',
        ready: false,
        database: 'connecting',
        dependencies: {
          gait: gaitHealth.data,
          therapy: therapyHealth.data
        }
      });
    }

    return res.json({
      success: true,
      service: 'node-api',
      ready: true,
      database: 'connected',
      dependencies: {
        gait: gaitHealth.data,
        therapy: therapyHealth.data
      }
    });
  } catch (error) {
    return res.status(503).json({
      success: false,
      service: 'node-api',
      ready: false,
      database: databaseReady ? 'connected' : 'connecting',
      error: isProduction ? 'Dependency health check failed' : error.message
    });
  }
});

// Error handler middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  const statusCode = err.statusCode || (err.message === 'Origin not allowed by CORS' ? 403 : 500);

  res.status(statusCode).json({
    success: false,
    message: isProduction && statusCode >= 500 ? 'Server Error' : (err.message || 'Server Error')
  });
});

const PORT = process.env.PORT || 5000;

let server;

async function startServer() {
  const db = await connectDB();
  app.locals.db = db;
  console.log('✅ Database instance made available to routes');

  server = app.listen(PORT, () => {
    console.log(`Server running in ${process.env.NODE_ENV} mode on port ${PORT}`);
  });
}

startServer().catch(error => {
  console.error('❌ Failed to start server:', error.message);
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
  console.log(`Error: ${err.message}`);
  // Close server & exit process
  if (server) {
    server.close(() => process.exit(1));
    return;
  }

  process.exit(1);
});
