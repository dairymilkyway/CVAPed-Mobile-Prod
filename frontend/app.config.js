const { expo } = require('./app.json');

const googleServicesFile = process.env.GOOGLE_SERVICES_JSON || './google-services.json';
const projectId = '01b0ad1a-a7a8-4d70-9b80-65c983dbda20';

module.exports = {
  expo: {
    ...expo,
    extra: {
      ...(expo.extra || {}),
      eas: {
        ...((expo.extra && expo.extra.eas) || {}),
        projectId,
      },
    },
    android: {
      ...expo.android,
      googleServicesFile,
    },
  },
};
