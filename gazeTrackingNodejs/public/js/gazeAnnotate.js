(() => {
  // —————————————— CONFIGURATION ——————————————
  // Time (ms) to consider as “hover threshold”
  const DWELL_TIME_MS = 3000; // 3 seconds

  // Radius threshold (px). If all gaze points in history are within this radius,
  // we consider it “focused on the same region.”
  const RADIUS_THRESHOLD = 50;

  // Canvas & context
  const canvas = document.getElementById("gazeCanvas");
  const ctx = canvas.getContext("2d");

  // Keep track of canvas size
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  // History of gaze points: each entry is { x: px, y: px, t: msSinceEpoch }
  let gazeHistory = [];

  let boxDrawn = false; // whether we’ve already triggered the prompt for the current dwell
  let currentBox = null; // store { xMin, yMin, xMax, yMax } of detected region

  // Draw a small circle at (x, y), and if boxDrawn, also draw the rectangle
  function drawOverlay(gazePointPx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw gaze dot (radius=8px, bright color)
    ctx.beginPath();
    ctx.arc(gazePointPx.x, gazePointPx.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#00FF00";
    ctx.fill();

    // If we have a bounding box to show, draw it
    if (boxDrawn && currentBox) {
      const { xMin, yMin, xMax, yMax } = currentBox;
      ctx.strokeStyle = "#FF0000";
      ctx.lineWidth = 3;
      ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
    }
  }

  // Check if all points in `pts` lie within RADIUS_THRESHOLD px of each other
  function allPointsWithinRadius(pts) {
    if (pts.length === 0) return false;
    // Compute mean center
    let sumX = 0,
      sumY = 0;
    pts.forEach((p) => {
      sumX += p.x;
      sumY += p.y;
    });
    const meanX = sumX / pts.length;
    const meanY = sumY / pts.length;
    // Verify each point is within RADIUS_THRESHOLD of that center
    return pts.every((p) => {
      const dx = p.x - meanX;
      const dy = p.y - meanY;
      return Math.hypot(dx, dy) <= RADIUS_THRESHOLD;
    });
  }

  // Given an array of points, compute the bounding box { xMin, yMin, xMax, yMax }
  function computeBoundingBox(pts) {
    let xMin = Infinity,
      yMin = Infinity,
      xMax = -Infinity,
      yMax = -Infinity;
    pts.forEach((p) => {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    });
    return { xMin, yMin, xMax, yMax };
  }

  // —————————————— WEBSOCKET SETUP ——————————————
  const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = `${wsProtocol}://${location.host}`;
  const socket = new WebSocket(wsUrl);

  socket.addEventListener("open", () => {
    console.log("WebSocket connected → waiting for gaze data…");
  });

  socket.addEventListener("message", (event) => {
    // Parse the JSON: { x: normalized, y: normalized, t: ms }
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.warn("Invalid JSON from server:", event.data);
      return;
    }
    const { x: normX, y: normY, t } = msg;
    // Convert normalized → pixel coordinates on the canvas
    const px = normX * canvas.width;
    const py = normY * canvas.height;

    // Add to gazeHistory
    gazeHistory.push({ x: px, y: py, t });
    // Remove any point older than DWELL_TIME_MS
    const cutoff = t - DWELL_TIME_MS;
    gazeHistory = gazeHistory.filter((pt) => pt.t >= cutoff);

    // Only attempt a dwell‐check if we’ve collected ≥ DWELL_TIME_MS worth of data
    if (!boxDrawn && gazeHistory.length > 0) {
      const earliestTimestamp = gazeHistory[0].t;
      if (t - earliestTimestamp >= DWELL_TIME_MS) {
        // We have a full 3 seconds of data in gazeHistory
        const pts3sec = gazeHistory.slice(); // all points in last 3s
        if (allPointsWithinRadius(pts3sec)) {
          // Compute bounding box of these points
          currentBox = computeBoundingBox(pts3sec);
          boxDrawn = true;

          // Draw the box immediately
          drawOverlay({ x: px, y: py });

          // Use built‐in confirm() instead of custom modal
          const userConfirmed = window.confirm("Correct annotation?");
          if (userConfirmed) {
            console.log(
              "User clicked OK → confirmed annotation for box:",
              currentBox
            );
          } else {
            console.log(
              "User clicked Cancel → rejecting annotation for box:",
              currentBox
            );
          }

          // Clear everything so future dwells can fire again
          boxDrawn = false;
          currentBox = null;
          gazeHistory = []; // optional: discard old points after a decision
        }
      }
    }

    // Always redraw the dot (and box if still flagged—but we cleared boxDrawn immediately)
    drawOverlay({ x: px, y: py });
  });

  socket.addEventListener("close", () => {
    console.log("WebSocket closed");
  });

  socket.addEventListener("error", (err) => {
    console.error("WebSocket error:", err);
  });
})();
