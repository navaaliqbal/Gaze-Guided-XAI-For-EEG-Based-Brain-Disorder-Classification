// server.js - parses any REC chunk for gaze, broadcasts normalized [0-1] coords
// Sliding-window (box) average
const net = require("net");
const express = require("express");
const http = require("http");
const WebSocket = require("ws");

// Server the static front-end
const app = express();
app.use(express.static("public"));
const server = http.createServer(app);

// WebSocket server
const wss = new WebSocket.Server({ server });
function broadcast(data) {
  const msg = JSON.stringify(data);
  wss.clients.forEach((ws) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(msg);
  });
}

// ─── SLIDING WINDOW SETUP ─────────────────────────────────
const windowSize = 5;
const bufX = new Array(windowSize).fill(0);
const bufY = new Array(windowSize).fill(0);
let buffPos = 0;
// Connect to GP3
let xmlBuffer = "";
const gp = new net.Socket();
gp.connect(4242, "100.87.72.127", () => {
  console.log("Connected to GP3");
  // must enable the global data stream first;
  gp.write('<SET ID="ENABLE_SEND_DATA" STATE="1"/>\r\n');
  // then enable the POG channels you want:
  gp.write('<SET ID="ENABLE_SEND_POG_BEST" STATE="1"/>\r\n');
  gp.write('<SET ID="ENABLE_SEND_POG_LEFT"> STATE="1"/>\r\n');
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
    // DEBUG: log raw xml
    console.log("RAW:", chunk.trim());

    // try to extrack normalized gaze: first left-eye, then best-eye
    let m =
      /LPOGX="([\d.]+)".*?LPOGY="([\d.]+)"/.exec(chunk) ||
      /BPOGX="([\d.]+)".*?BPOGY="([\d.]+)"/.exec(chunk);

    if (m) {
      const nx = parseFloat(m[1]); // normalized [0-1]
      const ny = parseFloat(m[2]);

      // ─── INSERT INTO RING BUFFER ───────────────────────────
      bufX[buffPos] = nx;
      bufY[buffPos] = ny;
      buffPos = (buffPos + 1) % windowSize;

      // ─── COMPUTE AVERAGE ─────────────────────
      const sumX = bufX.reduce((a, b) => a + b, 0);
      const sumY = bufY.reduce((a, b) => a + b, 0);
      const avgX = sumX / windowSize;
      const avgY = sumY / windowSize;
      // ─────────────────────────────────────────
      console.log(
        `gaze raw x=${nx.toFixed(3)}, y=${ny.toFixed(3)} →` +
          `avg[${windowSize}] x=${avgX.toFixed(3)}, y=${avgY.toFixed(3)}`
      );
      broadcast({ x: avgX, y: avgY });
    }
  }
});

gp.on("error", (err) => console.log("GP3 socket error", err.message));
gp.on("close", () => {
  console.warn("GP3 socket closed - reconnecting in 3s");
  setTimeout(() => gp.connect(4242, "100.87.72.127"), 3000);
});

const PORT = 3000;
server.listen(PORT, () => {
  console.log(`HTTP + WS server listening on 100.87.72.127:${PORT}`);
});
