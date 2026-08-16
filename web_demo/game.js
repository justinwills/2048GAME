(function () {
  "use strict";

  const SIZE = 4;
  const WIN_TILE = 2048;
  const BEST_KEY = "2048-best-score";
  const ANIM_MS = 150;

  let cells = [];
  let score = 0;
  let bestScore = parseInt(localStorage.getItem(BEST_KEY) || "0", 10);
  let won = false;
  let keepPlaying = false;
  let gameOver = false;
  let animating = false;
  let autoPlay = false;
  let aiTimer = null;
  let nextTileId = 0;

  const tileEls = new Map();

  const gridBackground = document.getElementById("grid-background");
  const tileContainer = document.getElementById("tile-container");
  const scoreEl = document.getElementById("score");
  const bestScoreEl = document.getElementById("best-score");
  const gameMessage = document.getElementById("game-message");
  const messageText = document.getElementById("message-text");
  const newGameBtn = document.getElementById("new-game");
  const autoPlayBtn = document.getElementById("auto-play");
  const tryAgainBtn = document.getElementById("try-again");
  const gameContainer = document.querySelector(".game-container");

  function emptyGrid() {
    return Array.from({ length: SIZE }, () => Array(SIZE).fill(null));
  }

  function Tile(value, r, c) {
    this.id = ++nextTileId;
    this.value = value;
    this.r = r;
    this.c = c;
  }

  function snapshotPositions(grid) {
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const tile = grid[r][c];
        if (tile) {
          tile.prevR = r;
          tile.prevC = c;
        }
      }
    }
  }

  function getEmptyCells(grid) {
    const list = [];
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (!grid[r][c]) list.push({ r, c });
      }
    }
    return list;
  }

  function addRandomTile(grid) {
    const empty = getEmptyCells(grid);
    if (!empty.length) return null;
    const { r, c } = empty[Math.floor(Math.random() * empty.length)];
    const tile = new Tile(Math.random() < 0.9 ? 2 : 4, r, c);
    grid[r][c] = tile;
    return tile;
  }

  function processLine(tiles) {
    const survivors = [];
    const toRemove = [];
    const mergedTiles = [];
    let gained = 0;
    let i = 0;

    while (i < tiles.length) {
      if (i + 1 < tiles.length && tiles[i].value === tiles[i + 1].value) {
        const survivor = tiles[i];
        const partner = tiles[i + 1];
        survivor.value *= 2;
        gained += survivor.value;
        mergedTiles.push(survivor);
        toRemove.push(partner);
        survivors.push(survivor);
        i += 2;
      } else {
        survivors.push(tiles[i]);
        i += 1;
      }
    }

    return { survivors, toRemove, mergedTiles, gained };
  }

  function assignMergeTargets(toRemove, mergedTiles) {
    for (const partner of toRemove) {
      const survivor = mergedTiles.find(
        (t) => t.prevR === partner.prevR && t.prevC === partner.prevC - 1
      );
      if (survivor) {
        partner.r = survivor.r;
        partner.c = survivor.c;
      }
    }
  }

  function placeLine(grid, row, survivors, startCol, step) {
    for (let c = 0; c < SIZE; c++) grid[row][c] = null;
    survivors.forEach((tile, index) => {
      const col = startCol + index * step;
      tile.r = row;
      tile.c = col;
      grid[row][col] = tile;
    });
  }

  function placeColumn(grid, col, survivors, startRow, step) {
    for (let r = 0; r < SIZE; r++) grid[r][col] = null;
    survivors.forEach((tile, index) => {
      const row = startRow + index * step;
      tile.r = row;
      tile.c = col;
      grid[row][col] = tile;
    });
  }

  function moveLeft(grid) {
    snapshotPositions(grid);
    const toRemove = [];
    const mergedTiles = [];
    let gained = 0;

    for (let r = 0; r < SIZE; r++) {
      const line = [];
      for (let c = 0; c < SIZE; c++) {
        if (grid[r][c]) line.push(grid[r][c]);
      }

      const result = processLine(line);
      gained += result.gained;
      toRemove.push(...result.toRemove);
      mergedTiles.push(...result.mergedTiles);
      placeLine(grid, r, result.survivors, 0, 1);
      assignMergeTargets(result.toRemove, result.mergedTiles);
    }

    return { moved: didMove(grid, toRemove), gained, toRemove, mergedTiles };
  }

  function moveRight(grid) {
    snapshotPositions(grid);
    const toRemove = [];
    const mergedTiles = [];
    let gained = 0;

    for (let r = 0; r < SIZE; r++) {
      const line = [];
      for (let c = SIZE - 1; c >= 0; c--) {
        if (grid[r][c]) line.push(grid[r][c]);
      }

      const result = processLine(line);
      gained += result.gained;
      toRemove.push(...result.toRemove);
      mergedTiles.push(...result.mergedTiles);
      placeLine(grid, r, result.survivors, SIZE - 1, -1);

      for (const partner of result.toRemove) {
        const survivor = result.mergedTiles.find(
          (t) => t.prevR === partner.prevR && t.prevC === partner.prevC + 1
        );
        if (survivor) {
          partner.r = survivor.r;
          partner.c = survivor.c;
        }
      }
    }

    return { moved: didMove(grid, toRemove), gained, toRemove, mergedTiles };
  }

  function moveUp(grid) {
    snapshotPositions(grid);
    const toRemove = [];
    const mergedTiles = [];
    let gained = 0;

    for (let c = 0; c < SIZE; c++) {
      const line = [];
      for (let r = 0; r < SIZE; r++) {
        if (grid[r][c]) line.push(grid[r][c]);
      }

      const result = processLine(line);
      gained += result.gained;
      toRemove.push(...result.toRemove);
      mergedTiles.push(...result.mergedTiles);
      placeColumn(grid, c, result.survivors, 0, 1);

      for (const partner of result.toRemove) {
        const survivor = result.mergedTiles.find(
          (t) => t.prevC === partner.prevC && t.prevR === partner.prevR - 1
        );
        if (survivor) {
          partner.r = survivor.r;
          partner.c = survivor.c;
        }
      }
    }

    return { moved: didMove(grid, toRemove), gained, toRemove, mergedTiles };
  }

  function moveDown(grid) {
    snapshotPositions(grid);
    const toRemove = [];
    const mergedTiles = [];
    let gained = 0;

    for (let c = 0; c < SIZE; c++) {
      const line = [];
      for (let r = SIZE - 1; r >= 0; r--) {
        if (grid[r][c]) line.push(grid[r][c]);
      }

      const result = processLine(line);
      gained += result.gained;
      toRemove.push(...result.toRemove);
      mergedTiles.push(...result.mergedTiles);
      placeColumn(grid, c, result.survivors, SIZE - 1, -1);

      for (const partner of result.toRemove) {
        const survivor = result.mergedTiles.find(
          (t) => t.prevC === partner.prevC && t.prevR === partner.prevR + 1
        );
        if (survivor) {
          partner.r = survivor.r;
          partner.c = survivor.c;
        }
      }
    }

    return { moved: didMove(grid, toRemove), gained, toRemove, mergedTiles };
  }

  function didMove(grid, toRemove) {
    if (toRemove.length > 0) return true;

    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const tile = grid[r][c];
        if (tile && (tile.r !== tile.prevR || tile.c !== tile.prevC)) return true;
      }
    }
    return false;
  }

  function canMoveGrid(grid) {
    if (getEmptyCells(grid).length > 0) return true;
    for (const fn of [moveLeft, moveRight, moveUp, moveDown]) {
      const copy = cloneGrid(grid);
      if (fn(copy).moved) return true;
    }
    return false;
  }

  function cloneGrid(grid) {
    const copy = emptyGrid();
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (grid[r][c]) copy[r][c] = new Tile(grid[r][c].value, r, c);
      }
    }
    return copy;
  }

  function maxTileValue(grid) {
    let max = 0;
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (grid[r][c] && grid[r][c].value > max) max = grid[r][c].value;
      }
    }
    return max;
  }

  const moves = {
    ArrowLeft: moveLeft,
    ArrowRight: moveRight,
    ArrowUp: moveUp,
    ArrowDown: moveDown,
    a: moveLeft,
    d: moveRight,
    w: moveUp,
    s: moveDown,
  };

  function buildGridBackground() {
    gridBackground.innerHTML = "";
    for (let i = 0; i < SIZE * SIZE; i++) {
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      gridBackground.appendChild(cell);
    }
  }

  function getCellRect(r, c) {
    const cell = gridBackground.children[r * SIZE + c];
    const boardRect = tileContainer.getBoundingClientRect();
    const cellRect = cell.getBoundingClientRect();
    return {
      x: cellRect.left - boardRect.left,
      y: cellRect.top - boardRect.top,
      size: cellRect.width,
    };
  }

  function setTileTransform(el, x, y, size, instant) {
    el.style.width = `${size}px`;
    el.style.height = `${size}px`;
    if (instant) el.classList.add("tile-no-transition");
    el.style.transform = `translate(${x}px, ${y}px)`;
    if (instant) {
      void el.offsetWidth;
      el.classList.remove("tile-no-transition");
    }
  }

  function applyTileAt(el, r, c, instant) {
    const { x, y, size } = getCellRect(r, c);
    setTileTransform(el, x, y, size, instant);
  }

  function applyTileStyle(el, tile, instant) {
    applyTileAt(el, tile.r, tile.c, instant);
  }

  function tileClass(value) {
    if (value <= 2048) return `tile-${value}`;
    return "tile-super";
  }

  function createTileElement(tile, isNew) {
    const el = document.createElement("div");
    el.className = `tile ${tileClass(tile.value)}`;
    if (isNew) el.classList.add("tile-new");

    const inner = document.createElement("div");
    inner.className = "tile-inner";
    inner.textContent = tile.value;
    el.appendChild(inner);

    applyTileStyle(el, tile, true);
    tileEls.set(tile.id, el);
    tileContainer.appendChild(el);
    return el;
  }

  function updateTileElement(tile, merged) {
    const el = tileEls.get(tile.id);
    if (!el) return;
    el.className = `tile ${tileClass(tile.value)}${merged ? " tile-merged" : ""}`;
    el.querySelector(".tile-inner").textContent = tile.value;
  }

  function removeTileElement(tile) {
    const el = tileEls.get(tile.id);
    if (el) {
      el.remove();
      tileEls.delete(tile.id);
    }
  }

  function clearAllTiles() {
    tileEls.forEach((el) => el.remove());
    tileEls.clear();
    tileContainer.innerHTML = "";
  }

  function positionAllTiles(instant) {
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const tile = cells[r][c];
        if (tile) {
          const el = tileEls.get(tile.id);
          if (el) applyTileStyle(el, tile, instant);
        }
      }
    }
  }

  function collectSlidingTiles(toRemove) {
    const sliding = [];
    const seen = new Set();

    for (const tile of toRemove) {
      if (!seen.has(tile.id)) {
        seen.add(tile.id);
        sliding.push(tile);
      }
    }

    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const tile = cells[r][c];
        if (tile && !seen.has(tile.id)) {
          seen.add(tile.id);
          sliding.push(tile);
        }
      }
    }

    return sliding;
  }

  function updateScores() {
    scoreEl.textContent = score;
    if (score > bestScore) {
      bestScore = score;
      localStorage.setItem(BEST_KEY, String(bestScore));
    }
    bestScoreEl.textContent = bestScore;
  }

  function showMessage(text) {
    messageText.textContent = text;
    tryAgainBtn.textContent = won && !gameOver ? "Keep playing" : "Try again";
    gameMessage.classList.remove("hidden");
  }

  function hideMessage() {
    gameMessage.classList.add("hidden");
  }

  function cellsToBoard() {
    const board = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        board[r][c] = cells[r][c] ? cells[r][c].value : 0;
      }
    }
    return board;
  }

  function clearAiTimer() {
    if (aiTimer !== null) {
      clearTimeout(aiTimer);
      aiTimer = null;
    }
  }

  function scheduleAiStep() {
    clearAiTimer();
    if (!autoPlay || animating || gameOver) return;
    if (won && !keepPlaying) return;

    aiTimer = setTimeout(() => {
      aiTimer = null;
      if (!autoPlay || animating || gameOver) return;

      const direction = window.AI2048.getBestArrow(cellsToBoard());
      if (!direction) return;

      const moved = performMove(direction);
      if (!moved && autoPlay) scheduleAiStep();
    }, ANIM_MS + 60);
  }

  function setAutoPlay(on) {
    autoPlay = on;
    autoPlayBtn.classList.toggle("is-active", on);
    autoPlayBtn.textContent = on ? "Stop AI" : "Auto Play";

    if (on) {
      keepPlaying = true;
      hideMessage();
      scheduleAiStep();
    } else {
      clearAiTimer();
    }
  }

  function checkGameState() {
    if (!won && maxTileValue(cells) >= WIN_TILE) {
      won = true;
      if (autoPlay) {
        keepPlaying = true;
        scheduleAiStep();
        return;
      }
      showMessage("You win!");
      return;
    }
    if (!canMoveGrid(cells)) {
      gameOver = true;
      if (autoPlay) setAutoPlay(false);
      showMessage("Game over!");
      return;
    }

    if (autoPlay) scheduleAiStep();
  }

  function finishMove(result, sliding) {
    for (const tile of sliding) {
      const el = tileEls.get(tile.id);
      if (el) el.classList.remove("tile-sliding");
    }

    for (const tile of result.toRemove) {
      removeTileElement(tile);
    }

    for (const tile of result.mergedTiles) {
      updateTileElement(tile, true);
      const el = tileEls.get(tile.id);
      if (el) applyTileStyle(el, tile, true);
    }

    const newTile = addRandomTile(cells);
    if (newTile) createTileElement(newTile, true);

    animating = false;
    checkGameState();
  }

  function performMove(direction) {
    const moveFn = moves[direction];
    if (!moveFn || animating || gameOver || (won && !keepPlaying)) return false;

    const result = moveFn(cells);
    if (!result.moved) return false;

    animating = true;
    score += result.gained;
    updateScores();

    const sliding = collectSlidingTiles(result.toRemove);

    for (const tile of sliding) {
      const el = tileEls.get(tile.id);
      if (!el) continue;
      el.classList.add("tile-sliding");
      applyTileAt(el, tile.prevR, tile.prevC, true);
    }

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        for (const tile of sliding) {
          const el = tileEls.get(tile.id);
          if (el) applyTileStyle(el, tile, false);
        }

        setTimeout(() => finishMove(result, sliding), ANIM_MS);
      });
    });

    return true;
  }

  function setupGame() {
    clearAiTimer();
    cells = emptyGrid();
    score = 0;
    won = false;
    keepPlaying = autoPlay;
    gameOver = false;
    animating = false;
    hideMessage();
    clearAllTiles();

    const t1 = addRandomTile(cells);
    const t2 = addRandomTile(cells);
    if (t1) createTileElement(t1, true);
    if (t2) createTileElement(t2, true);

    updateScores();

    if (autoPlay) scheduleAiStep();
  }

  function onKeyDown(e) {
    const key = e.key;
    if (autoPlay && (key in moves || key.startsWith("Arrow"))) {
      e.preventDefault();
      return;
    }
    if (key in moves || key.startsWith("Arrow")) {
      e.preventDefault();
      if (key === "r" || key === "R") {
        setupGame();
        return;
      }
      performMove(key);
    }
  }

  let touchStartX = 0;
  let touchStartY = 0;

  function onTouchStart(e) {
    if (e.touches.length !== 1) return;
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  }

  function onTouchEnd(e) {
    if (autoPlay) return;
    if (!touchStartX && !touchStartY) return;

    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    if (Math.max(absDx, absDy) < 30) return;

    if (absDx > absDy) {
      performMove(dx > 0 ? "ArrowRight" : "ArrowLeft");
    } else {
      performMove(dy > 0 ? "ArrowDown" : "ArrowUp");
    }

    touchStartX = 0;
    touchStartY = 0;
  }

  tryAgainBtn.addEventListener("click", () => {
    if (won && !keepPlaying && !gameOver) {
      keepPlaying = true;
      hideMessage();
      return;
    }
    setupGame();
  });

  newGameBtn.addEventListener("click", setupGame);
  autoPlayBtn.addEventListener("click", () => setAutoPlay(!autoPlay));
  document.addEventListener("keydown", onKeyDown);
  gameContainer.addEventListener("touchstart", onTouchStart, { passive: true });
  gameContainer.addEventListener("touchend", onTouchEnd, { passive: true });
  window.addEventListener("resize", () => {
    if (!animating) positionAllTiles(true);
  });

  buildGridBackground();
  bestScoreEl.textContent = bestScore;
  requestAnimationFrame(setupGame);
})();
