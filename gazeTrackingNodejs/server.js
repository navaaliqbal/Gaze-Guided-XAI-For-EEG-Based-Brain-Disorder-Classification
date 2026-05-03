// server.js

const net = require("net");
const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const fs = require('fs').promises;
const path = require('path');

// Server the static front-end
const app = express();
app.use(express.static("public"));
app.use(express.json());
const server = http.createServer(app);

// WebSocket server
const wss = new WebSocket.Server({ server });
function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((ws) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(msg);
  });
}

// Create annotations directory if it doesn't exist
const ANNOTATIONS_DIR = path.join(__dirname, 'annotations');
fs.mkdir(ANNOTATIONS_DIR, { recursive: true }).catch(console.error);

// Gaze Stabilization Configuration
const KALMAN_CONFIG = {
    processNoise: 0.001,    // Lower = more smoothing but more lag
    measurementNoise: 0.1,  // Higher = trust measurements less
    estimateError: 1,      // Initial estimate error
    windowSize: 10         // Moving average window size
};

// Kalman filter implementation for each axis
class KalmanFilter {
    constructor() {
        this.processNoise = KALMAN_CONFIG.processNoise;
        this.measurementNoise = KALMAN_CONFIG.measurementNoise;
        this.estimateError = KALMAN_CONFIG.estimateError;
        this.lastEstimate = 0;
        this.lastError = this.estimateError;
    }

    update(measurement) {
        // Prediction
        const predictedEstimate = this.lastEstimate;
        const predictedError = this.lastError + this.processNoise;

        // Update
        const kalmanGain = predictedError / (predictedError + this.measurementNoise);
        const currentEstimate = predictedEstimate + kalmanGain * (measurement - predictedEstimate);
        const currentError = (1 - kalmanGain) * predictedError;

        // Save for next iteration
        this.lastEstimate = currentEstimate;
        this.lastError = currentError;

        return currentEstimate;
    }
}

// Moving average buffer for additional smoothing
class MovingAverageFilter {
    constructor(size) {
        this.values = [];
        this.size = size;
    }

    update(value) {
        this.values.push(value);
        if (this.values.length > this.size) {
            this.values.shift();
        }
        return this.values.reduce((a, b) => a + b, 0) / this.values.length;
    }
}

// Create filters for X and Y coordinates
const kalmanX = new KalmanFilter();
const kalmanY = new KalmanFilter();
const avgX = new MovingAverageFilter(KALMAN_CONFIG.windowSize);
const avgY = new MovingAverageFilter(KALMAN_CONFIG.windowSize);

// Velocity-based outlier detection
let lastX = null;
let lastY = null;
let lastTime = null;
const MAX_VELOCITY = 1.0; // maximum allowed velocity in screen units per millisecond

let xmlBuffer = "";
const gp = new net.Socket();
gp.connect(4242, "100.87.72.127", () => {
  console.log("Connected to GP3");
  gp.write('<SET ID="ENABLE_SEND_DATA" STATE="1"/>\r\n');
  gp.write('<SET ID="ENABLE_SEND_POG_BEST" STATE="1"/>\r\n');
  gp.write('<SET ID="ENABLE_SEND_POG_LEFT" STATE="1"/>\r\n');
  gp.write("<SET ID='ENABLE_SEND_POG_RIGHT' STATE='1'/>\r\n");
  gp.write('<SET ID="ENABLE_SEND_TIME_SYNC" STATE="1"/>\r\n');
  console.log("Sent ENABLE_SEND_DATA + POG/Time commands");
});

gp.on("data", (buf) => {
  xmlBuffer += buf.toString();
  const parts = xmlBuffer.split(">");
  xmlBuffer = parts.pop();

  for (let chunk of parts) {
    chunk = chunk + ">";
    // console.log("RAW:", chunk.trim());

    // extract normalized gaze: left‐eye or best‐eye
    const m =
      /LPOGX="([\d.]+)".*?LPOGY="([\d.]+)"/.exec(chunk) ||
      /BPOGX="([\d.]+)".*?BPOGY="([\d.]+)"/.exec(chunk);

    if (m) {
      const nx = parseFloat(m[1]); // normalized [0–1]
      const ny = parseFloat(m[2]); // normalized [0–1]

      // ─── APPLY MOVING‐AVERAGE SMOOTHING ──────────────────
      // Push raw normalized values into history
      pushGazeSample(gazeHistoryX, nx);
      pushGazeSample(gazeHistoryY, ny);

      // Compute average of the last WINDOW_SIZE samples
      const smoothX = average(gazeHistoryX);
      const smoothY = average(gazeHistoryY);
      // ─────────────────────────────────────────────────────

      console.log(
        `gaze raw x=${nx.toFixed(3)}, y=${ny.toFixed(3)} → ` +
          `smoothed x=${smoothX.toFixed(3)}, y=${smoothY.toFixed(3)}`
      );
      broadcast({ x: smoothX, y: smoothY });
    }
  }
});

gp.on("error", (err) => console.log("GP3 socket error", err.message));
gp.on("close", () => {
  console.warn("GP3 socket closed - reconnecting in 3s");
  setTimeout(() => gp.connect(4242, "100.87.72.127"), 3000);
});

// Save annotations endpoint
app.post('/save-annotations', async (req, res) => {
    try {
        const { fileName, annotations } = req.body;
        if (!fileName || !annotations) {
            return res.status(400).json({ error: 'Missing required fields' });
        }

        const filePath = path.join(ANNOTATIONS_DIR, fileName);
        await fs.writeFile(filePath, JSON.stringify(annotations, null, 2));
        res.json({ success: true });
    } catch (error) {
        console.error('Error saving annotations:', error);
        res.status(500).json({ error: 'Failed to save annotations' });
    }
});

// Load annotations endpoint
app.get('/load-annotations', async (req, res) => {
    try {
        const fileName = req.query.file;
        if (!fileName) {
            return res.status(400).json({ error: 'Missing file name' });
        }

        const filePath = path.join(ANNOTATIONS_DIR, fileName);
        const data = await fs.readFile(filePath, 'utf8');
        res.json(JSON.parse(data));
    } catch (error) {
        if (error.code === 'ENOENT') {
            // File doesn't exist yet - return empty array
            return res.json([]);
        }
        console.error('Error loading annotations:', error);
        res.status(500).json({ error: 'Failed to load annotations' });
    }
});

// WebSocket connection handling for gaze data
wss.on('connection', function connection(ws) {
    console.log('New WebSocket connection');
    
    ws.on('message', function incoming(message) {
        try {
            const data = JSON.parse(message);
            const now = Date.now();
            
            // Apply Kalman filter
            let stabilizedX = kalmanX.update(data.x);
            let stabilizedY = kalmanY.update(data.y);
            
            // Apply moving average
            stabilizedX = avgX.update(stabilizedX);
            stabilizedY = avgY.update(stabilizedY);
            
            // Velocity-based outlier detection
            if (lastX !== null && lastTime !== null) {
                const dt = now - lastTime;
                const vx = Math.abs(stabilizedX - lastX) / dt;
                const vy = Math.abs(stabilizedY - lastY) / dt;
                
                if (vx > MAX_VELOCITY || vy > MAX_VELOCITY) {
                    // Skip this point if velocity is too high
                    return;
                }
            }
            
            // Update last known position
            lastX = stabilizedX;
            lastY = stabilizedY;
            lastTime = now;
            
            // Broadcast stabilized coordinates
            const stabilizedData = JSON.stringify({
                x: stabilizedX,
                y: stabilizedY,
                t: now
            });
            
            wss.clients.forEach(function each(client) {
                if (client.readyState === WebSocket.OPEN) {
                    client.send(stabilizedData);
                }
            });
        } catch (error) {
            console.error('Error processing gaze data:', error);
        }
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
