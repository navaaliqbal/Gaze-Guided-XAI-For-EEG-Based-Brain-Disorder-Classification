// drawRect.js
// ---------------------------
// This script replaces mouse‐based rectangle selection with gaze‐based selection.
// It opens a WebSocket to receive normalized gaze points from GP3, detects two consecutive fixations,
// draws a rectangle between those fixation points, and then runs your original “mouseup” logic
// (time mapping, channel selection, CSV export, etc.) unchanged.
// ---------------------------

// === 1. Canvas Setup (same as your original) ===
const canvas = document.createElement("canvas");
canvas.style.width = "100%";
canvas.style.height = "100%";
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
canvas.style.position = "absolute";
canvas.style.left = "0";
canvas.style.top = "0";
canvas.style.zIndex = "100001";
canvas.style.pointerEvents = "none"; // allow clicks/interaction “through” the canvas
document.body.appendChild(canvas);

const ctx = canvas.getContext("2d");

// Keep canvas sized to window
window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

// The container in which your EEG channels (div.ChDiv) live:
const div = document.getElementById("mycont");

// === 2. CSV & Modal Logic (unchanged from your original) ===
let csv_ar = [
  [
    "Gender",
    "Age",
    "File Start",
    "Start time",
    "End time",
    "Channel names",
    "Comment",
  ],
];
let com = "No comment";
let chArr = [];
let myStartnew = "";

// Modal & CSV helper functions (copied verbatim, except minor fixes to gen radio buttons)
const modal = document.getElementById("writeModal");
const button = document.getElementById("acceptWrite");
document.getElementById("write").onclick = () => {
  modalWrite();
};
document.getElementById("erase").onclick = () => {
  erase();
};

function modalWrite() {
  modal.style.display = "block";
}

button.onclick = () => {
  const us_age = $('input[name="getAge"]').val();
  let us_gen = "";
  if (document.getElementById("gen1")?.checked) {
    us_gen = document.getElementById("gen1").value;
  } else if (document.getElementById("gen2")?.checked) {
    us_gen = document.getElementById("gen2").value;
  } else if (document.getElementById("gen3")?.checked) {
    us_gen = document.getElementById("gen3").value;
  }
  modal.style.display = "none";
  csv_ar[1][0] = us_gen;
  csv_ar[1][1] = us_age;
  writeToCSV(csv_ar);
};

function writeToCSV(ar) {
  ar[1][2] = myStartnew;
  const csvContent =
    "data:text/csv;charset=utf-8," + ar.map((e) => e.join(",")).join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "my_data.csv");
  document.body.appendChild(link);
  link.click();
  erase();
}

window.onclick = (event) => {
  if (event.target == modal) {
    modal.style.display = "none";
  }
};

function erase() {
  if (window.confirm("Erase current labels?")) {
    csv_ar = [
      [
        "Gender",
        "Age",
        "File Start",
        "Start time",
        "End time",
        "Channel names",
        "Comment",
      ],
    ];
    // Optional: if you have a flag `call`, you can reset it here
    // call = 'e';
  }
}

// Placeholder stub for annotate_local & true_local (as in your original):
// If your main interface already includes these, you can omit this block.
let parsedata = [],
  trudata = [],
  first_draw_new = false;
let trufirst, trusecond, t_col1, t_col2, fs_col, tru_fs_new;

document
  .getElementById("ann-input")
  ?.addEventListener("change", annotate_local, false);
function annotate_local(e) {
  const ann_csv = e.target.files[0];
  const reader = new FileReader();
  reader.addEventListener("load", (ev) => {
    const csvdata = ev.target.result;
    const newLinebrk = csvdata.split("\n");
    for (let i = 0; i < newLinebrk.length; i++) {
      parsedata.push(newLinebrk[i].split(","));
    }
  });
  reader.readAsText(ann_csv);
  if (first_draw_new) {
    readEEG();
  }
}

document
  .getElementById("tru-input")
  ?.addEventListener("change", true_local, false);
function true_local(e) {
  console.log("Process trudata");
  const tru_csv = e.target.files[0];
  const reader = new FileReader();
  reader.addEventListener("load", (ev) => {
    const csvdata = ev.target.result;
    const newLinebrk = csvdata.split("\n");
    for (let i = 0; i < newLinebrk.length; i++) {
      trudata.push(newLinebrk[i].split(","));
    }
    trufirst = trudata[0];
    trusecond = trudata[1];
    t_col1 = trufirst.indexOf("Start time");
    t_col2 = trufirst.indexOf("End time");
    fs_col = trufirst.indexOf("File Start");
    trudata = trudata.map((val) => val.slice(t_col1, t_col2 + 1));
    trudata.shift();
    if (fs_col !== -1) {
      tru_fs_new = trusecond[fs_col];
    }
  });
  reader.readAsText(tru_csv);
  if (first_draw_new) {
    readEEG();
  }
}

// === 3. Fixation Detection & Gaze‐Based Rectangle Logic ===

// State management for two‐corner fixation sequence
let state = "WAITING_FOR_FIRST"; // or "FIXATED_FIRST"

// Buffers recent gaze samples: { x_abs, y_abs, t }
const gazeSamples = [];
const fixationThresholdPx = 30; // maximum dispersion in pixels
const fixationDurationMs = 600; // must hold within threshold for this long

let firstFixation = null; // { x: absoluteX, y: absoluteY }
let secondFixation = null; // { x: absoluteX, y: absoluteY }

// Helper: remove samples older than fixationDurationMs
function pruneOldSamples(now) {
  const windowStart = now - fixationDurationMs;
  while (gazeSamples.length && gazeSamples[0].t < windowStart) {
    gazeSamples.shift();
  }
}

// Helper: check if current gazeSamples form a fixation (low dispersion + enough duration)
function checkDispersionFixation() {
  if (gazeSamples.length < 2) return false;
  const earliest = gazeSamples[0].t;
  const latest = gazeSamples[gazeSamples.length - 1].t;
  const span = latest - earliest;
  if (span < fixationDurationMs) return false;

  let minX = Infinity,
    maxX = -Infinity;
  let minY = Infinity,
    maxY = -Infinity;
  gazeSamples.forEach((pt) => {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  });
  const dispX = maxX - minX;
  const dispY = maxY - minY;
  return dispX <= fixationThresholdPx && dispY <= fixationThresholdPx;
}

// Draw a small circle at a fixation point for user feedback
function drawFixationMarker(px, py, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px, py, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Once second fixation is locked, draw rectangle and run original “mouseup” logic:
function processRectangleFromFixations() {
  if (!firstFixation || !secondFixation) return;

  // Clear any previous drawings:
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Determine top-left corner (x0,y0) and width/height
  const x0 = Math.min(firstFixation.x, secondFixation.x);
  const y0 = Math.min(firstFixation.y, secondFixation.y);
  const w = Math.abs(secondFixation.x - firstFixation.x);
  const h = Math.abs(secondFixation.y - firstFixation.y);

  // Draw translucent rectangle
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.strokeStyle = "black";
  ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(150, 0, 0, 0.3)";
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  // Now run the original “mouseup” code, adapted to use these two fixations:

  // Get start‐corner absolute coords and end‐corner absolute coords
  const lm_abs_x = firstFixation.x;
  const lm_abs_y = firstFixation.y;
  const mouse_abs_x = secondFixation.x;
  const mouse_abs_y = secondFixation.y;

  // Read time window elements
  let tstart = document.getElementById("startWindowtime").innerHTML;
  const tend = document.getElementById("endWindowtime").innerHTML;
  const dur = parseInt(document.getElementById("wd").innerHTML, 10);
  let tsec = parseFloat(tstart.charAt(6) + tstart.charAt(7));
  tstart = tstart.substring(0, 6);

  // Compute coordinates relative to our container div
  const divRect = div.getBoundingClientRect();
  const relA_x = lm_abs_x - divRect.left;
  const relB_x = mouse_abs_x - divRect.left;

  // TIME‐MAPPING (exactly your original logic)
  let rect_start, rect_end;
  const a = dur * (relA_x / div.offsetWidth);
  if (tsec + a < 10) {
    rect_start = tstart + "0" + (tsec + a);
  } else {
    rect_start = tstart + (tsec + a);
  }

  const b = dur * (relB_x / div.offsetWidth);
  if (tsec + b < 10) {
    rect_end = tstart + "0" + (tsec + b);
  } else {
    rect_end = tstart + (tsec + b);
  }

  rect_start = rect_start.substring(0, 8) + ":" + rect_start.substring(9);
  rect_end = rect_end.substring(0, 8) + ":" + rect_end.substring(9);
  rect_start = rect_start.substring(0, 12);
  rect_end = rect_end.substring(0, 12);

  // Ensure start ≥ tsec
  const startSec = parseFloat(rect_start.charAt(6) + rect_start.charAt(7));
  if (startSec < tsec || (startSec === tsec && rect_start.length < 10)) {
    rect_start = rect_start.substring(0, 6) + tsec + ":000";
  }

  // CHANNEL SELECTION (your original loop over div.ChDiv)
  chArr = [];
  const topY = y0;
  const bottomY = y0 + h;
  $("div.ChDiv").each(function () {
    const r = this.getBoundingClientRect();
    if (
      (r.top >= topY && r.bottom <= bottomY) ||
      (r.top <= topY && r.bottom >= topY) ||
      (r.top >= topY && r.top <= bottomY) ||
      (r.bottom >= topY && r.bottom <= bottomY)
    ) {
      const chName = $(this).find("span").text();
      chArr.push(chName);
    }
  });

  // Only prompt if valid region & channels exist
  if (rect_start !== rect_end && chArr.length !== 0) {
    if (
      window.confirm(
        "Save selected region?\r\n" +
          "Number of channels = " +
          chArr.length +
          "\r\n" +
          "Start: " +
          rect_start +
          "\r\n" +
          "End:   " +
          rect_end
      )
    ) {
      com = window.prompt("Comment about abnormality", "No Comment");
      let s = chArr[0];
      for (let i = 1; i < chArr.length; i++) {
        s = s.concat(" ", chArr[i]);
      }

      // Convert HH:MM:SS:ms strings into zero‐padded components
      let st_mil = rect_start.substring(9);
      let end_mil = rect_end.substring(9);

      let st_h = parseInt(rect_start.substring(0, 2), 10);
      let end_h = parseInt(rect_end.substring(0, 2), 10);
      let st_m = parseInt(rect_start.substring(3, 5), 10);
      let end_m = parseInt(rect_end.substring(3, 5), 10);
      let st_s = parseInt(rect_start.substring(6, 8), 10);
      let end_s = parseInt(rect_end.substring(6, 8), 10);

      if (st_s > 59) {
        st_s -= 60;
        st_m += 1;
      }
      if (end_s > 59) {
        end_s -= 60;
        end_m += 1;
      }
      if (st_m > 59) {
        st_m -= 60;
        st_h += 1;
      }
      if (end_m > 59) {
        end_m -= 60;
        end_h += 1;
      }
      if (st_h > 23) {
        st_h -= 24;
      }
      if (end_h > 23) {
        end_h -= 24;
      }

      if (st_h < 10) {
        st_h = "0" + st_h;
        st_h = st_h.substring(0, 2);
      }
      if (st_m < 10) {
        st_m = "0" + st_m;
        st_m = st_m.substring(0, 2);
      }
      if (st_s < 10) {
        st_s = "0" + st_s;
        st_s = st_s.substring(0, 2);
      }
      if (end_h < 10) {
        end_h = "0" + end_h;
        end_h = end_h.substring(0, 2);
      }
      if (end_m < 10) {
        end_m = "0" + end_m;
        end_m = end_m.substring(0, 2);
      }
      if (end_s < 10) {
        end_s = "0" + end_s;
        end_s = end_s.substring(0, 2);
      }

      rect_start = st_h + ":" + st_m + ":" + st_s + ":" + st_mil;
      rect_end = end_h + ":" + end_m + ":" + end_s + ":" + end_mil;
      const new_row = ["", "", "", rect_start, rect_end, s, com];
      csv_ar.push(new_row);
    } else {
      // User canceled → clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  } else {
    alert("No region was selected. Please try again.");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // Reset for next two‐fixation cycle
  firstFixation = null;
  secondFixation = null;
  state = "WAITING_FOR_FIRST";
  gazeSamples.length = 0;
  chArr = [];
}

// === 4. WebSocket + Gaze Handling ===
// The server (server.js) is already streaming: { x: smoothX, y: smoothY } in [0..1].
// We open a WebSocket connection to receive those normalized gaze points.

const ws = new WebSocket(`ws://${location.host}`);
ws.addEventListener("message", (evt) => {
  const { x, y } = JSON.parse(evt.data); // normalized gaze in [0..1]

  // Convert normalized → absolute screen pixels
  const absX = x * window.innerWidth;
  const absY = y * window.innerHeight;
  const now = performance.now();

  // Push new gaze sample into our buffer
  gazeSamples.push({ x: absX, y: absY, t: now });
  pruneOldSamples(now);

  if (state === "WAITING_FOR_FIRST") {
    if (checkDispersionFixation()) {
      // First fixation detected
      // Compute centroid
      let sumX = 0,
        sumY = 0;
      gazeSamples.forEach((pt) => {
        sumX += pt.x;
        sumY += pt.y;
      });
      const cx = sumX / gazeSamples.length;
      const cy = sumY / gazeSamples.length;

      firstFixation = { x: cx, y: cy };
      drawFixationMarker(cx, cy, "green");

      state = "FIXATED_FIRST";
      gazeSamples.length = 0; // clear buffer for second fixation
    }
  } else if (state === "FIXATED_FIRST") {
    if (checkDispersionFixation()) {
      // Second fixation detected
      let sumX = 0,
        sumY = 0;
      gazeSamples.forEach((pt) => {
        sumX += pt.x;
        sumY += pt.y;
      });
      const cx = sumX / gazeSamples.length;
      const cy = sumY / gazeSamples.length;
      console.log(cx, cy);

      secondFixation = { x: cx, y: cy };
      drawFixationMarker(cx, cy, "green");

      state = "FIXATED_SECOND";

      // Delay slightly so user sees the second green dot before rectangle appears
      setTimeout(processRectangleFromFixations, 100);
    }
  }
});

// === 5. End of drawRect.js ===

// At this point, you can simply include <script src="drawRect.js"></script> in your main HTML,
// after ensuring that elements with IDs `mycont`, `startWindowtime`, `endWindowtime`, `wd`, `writeModal`,
// `acceptWrite`, and radio buttons `gen1`,`gen2`,`gen3` exist in your interface. The code will
// automatically connect to your GP3‐WebSocket server, detect two fixations, draw the rectangle, and run
// the original annotation logic (including CSV export).
//
// Note: You can adjust `fixationThresholdPx` (pixels) and `fixationDurationMs` (milliseconds) to tune
// how strict the fixation detection must be. Higher duration → user must hold gaze longer; smaller threshold → gaze must be steadier.
//
// This file does not include any pointer‐animation logic; it only deals with fixations → rectangles. If you want a live pointer dot, keep your separate gazePointer.js loaded as well.
//
