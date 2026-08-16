(function () {
  "use strict";

  const SIZE = 4;

  const weights = {
    empty: 1284.8,
    biggest: 0.5,
    smooth: 0.9,
    corner: 26.2,
    mono: 20.0,
  };

  const moveFns = {
    w: moveUp,
    s: moveDown,
    a: moveLeft,
    d: moveRight,
  };

  const moveToArrow = {
    w: "ArrowUp",
    s: "ArrowDown",
    a: "ArrowLeft",
    d: "ArrowRight",
  };

  function createBoard() {
    return Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
  }

  function copyBoard(board) {
    return board.map((row) => row.slice());
  }

  function boardsEqual(a, b) {
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (a[r][c] !== b[r][c]) return false;
      }
    }
    return true;
  }

  function moveRowLeft(row) {
    const filtered = row.filter((x) => x !== 0);
    const result = [];
    let gained = 0;
    let i = 0;

    while (i < filtered.length) {
      if (i + 1 < filtered.length && filtered[i] === filtered[i + 1]) {
        const merged = filtered[i] * 2;
        result.push(merged);
        gained += merged;
        i += 2;
      } else {
        result.push(filtered[i]);
        i += 1;
      }
    }

    while (result.length < SIZE) result.push(0);
    return { row: result, gained };
  }

  function moveLeft(board) {
    const newBoard = [];
    let totalGained = 0;
    for (const row of board) {
      const { row: newRow, gained } = moveRowLeft(row);
      newBoard.push(newRow);
      totalGained += gained;
    }
    return { board: newBoard, gained: totalGained };
  }

  function reverseBoard(board) {
    return board.map((row) => row.slice().reverse());
  }

  function transpose(board) {
    return Array.from({ length: SIZE }, (_, c) => board.map((row) => row[c]));
  }

  function moveRight(board) {
    const reversed = reverseBoard(board);
    const { board: moved, gained } = moveLeft(reversed);
    return { board: reverseBoard(moved), gained };
  }

  function moveUp(board) {
    const transposed = transpose(board);
    const { board: moved, gained } = moveLeft(transposed);
    return { board: transpose(moved), gained };
  }

  function moveDown(board) {
    const transposed = transpose(board);
    const { board: moved, gained } = moveRight(transposed);
    return { board: transpose(moved), gained };
  }

  function canMove(board) {
    if (board.some((row) => row.includes(0))) return true;
    for (const fn of Object.values(moveFns)) {
      if (!boardsEqual(board, fn(board).board)) return true;
    }
    return false;
  }

  function getEmptyCells(board) {
    const cells = [];
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (board[r][c] === 0) cells.push([r, c]);
      }
    }
    return cells;
  }

  function countEmpty(board) {
    let count = 0;
    for (const row of board) count += row.filter((v) => v === 0).length;
    return count;
  }

  function maxTile(board) {
    return Math.max(...board.flat());
  }

  function smoothness(board) {
    let penalty = 0;
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        if (board[r][c] === 0) continue;
        if (c + 1 < SIZE && board[r][c + 1] !== 0) {
          penalty += Math.abs(board[r][c] - board[r][c + 1]);
        }
        if (r + 1 < SIZE && board[r + 1][c] !== 0) {
          penalty += Math.abs(board[r][c] - board[r + 1][c]);
        }
      }
    }
    return -penalty;
  }

  function cornerBonus(board) {
    const biggest = maxTile(board);
    return board[0][0] === biggest ? biggest * 20 : 0;
  }

  function cornerLockedBonus(board) {
    const biggest = maxTile(board);
    const corners = [board[0][0], board[0][3], board[3][0], board[3][3]];
    return corners.includes(biggest) ? biggest * 30 : 0;
  }

  function monotonicity(board) {
    let total = 0;
    for (const row of board) {
      for (let i = 0; i < SIZE - 1; i++) {
        if (row[i] >= row[i + 1]) total += row[i];
      }
    }
    for (let c = 0; c < SIZE; c++) {
      for (let r = 0; r < SIZE - 1; r++) {
        if (board[r][c] >= board[r + 1][c]) total += board[r][c];
      }
    }
    return total;
  }

  function evaluate(board) {
    const empty = countEmpty(board);
    const biggest = maxTile(board);
    const smooth = smoothness(board);
    const corner = cornerBonus(board);
    const locked = cornerLockedBonus(board);
    const mono = monotonicity(board);

    let highTileBonus = 0;
    if (biggest >= 1024) highTileBonus += biggest * 50;
    if (biggest >= 2048) highTileBonus += biggest * 200;

    return (
      empty * weights.empty +
      biggest * weights.biggest +
      smooth * weights.smooth +
      corner * weights.corner +
      locked +
      mono * weights.mono +
      highTileBonus
    );
  }

  function expectimax(board, depth, isPlayerTurn) {
    if (depth === 0 || !canMove(board)) return evaluate(board);

    if (isPlayerTurn) {
      let bestScore = -1e15;

      for (const fn of Object.values(moveFns)) {
        const { board: newBoard, gained } = fn(board);
        if (boardsEqual(newBoard, board)) continue;

        const moveScore = gained + expectimax(newBoard, depth - 1, false);
        if (moveScore > bestScore) bestScore = moveScore;
      }

      return bestScore;
    }

    const emptyCells = getEmptyCells(board);
    if (!emptyCells.length) return evaluate(board);

    let totalScore = 0;
    for (const [r, c] of emptyCells) {
      const board2 = copyBoard(board);
      board2[r][c] = 2;
      totalScore += 0.9 * expectimax(board2, depth - 1, true);

      const board4 = copyBoard(board);
      board4[r][c] = 4;
      totalScore += 0.1 * expectimax(board4, depth - 1, true);
    }

    return totalScore / emptyCells.length;
  }

  function aiMove(board, depth) {
    let bestScore = -1e15;
    let bestMove = null;

    for (const [key, fn] of Object.entries(moveFns)) {
      const { board: newBoard, gained } = fn(board);
      if (boardsEqual(newBoard, board)) continue;

      const moveScore = gained + expectimax(newBoard, depth - 1, false);
      if (moveScore > bestScore) {
        bestScore = moveScore;
        bestMove = key;
      }
    }

    return bestMove;
  }

  function chooseDepth(board) {
    const biggest = maxTile(board);
    if (biggest >= 1024) return 5;
    if (biggest >= 512) return 4;
    return 3;
  }

  function getBestArrow(board) {
    const depth = chooseDepth(board);
    const key = aiMove(board, depth);
    return key ? moveToArrow[key] : null;
  }

  window.AI2048 = {
    getBestArrow,
    chooseDepth,
    createBoard,
  };
})();
