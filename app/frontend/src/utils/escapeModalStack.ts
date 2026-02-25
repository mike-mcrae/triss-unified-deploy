let listenersReady = false;
let nextId = 1;
const closeStack: Array<{ id: number; close: () => void }> = [];

function ensureListeners() {
  if (listenersReady) return;
  listenersReady = true;
  window.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    const top = closeStack[closeStack.length - 1];
    if (!top) return;
    event.preventDefault();
    top.close();
  });
}

export function registerEscapeClose(close: () => void): () => void {
  ensureListeners();
  const id = nextId++;
  closeStack.push({ id, close });
  return () => {
    const idx = closeStack.findIndex((item) => item.id === id);
    if (idx >= 0) closeStack.splice(idx, 1);
  };
}
