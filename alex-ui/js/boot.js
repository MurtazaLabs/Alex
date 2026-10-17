/**
 * Alex OS — boot marker.
 * Shell.js handles DOMContentLoaded wiring (auth gate + router start) and sets
 * window.__alexBooted, which the watchdog in index.html checks. This file is
 * the last script: if it executes, every script above loaded and parsed.
 */
(function () {
  'use strict';
  if (location.protocol === 'file:') {
    window.__bootFail && window.__bootFail('Alex UI cannot run from a file. Open http://localhost:8000');
    return;
  }
  // If shell.js somehow never ran (script error above this file), fail loudly.
  setTimeout(() => {
    if (!window.__alexBooted && window.__bootFail) {
      window.__bootFail('The app shell did not initialize — likely a script error. Open DevTools (F12) → Console and check for red text.');
    }
  }, 3000);
})();
