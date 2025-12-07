const COLLECTOR_DB_NAME = 'nodoCollectorDB';
const COLLECTOR_DB_VERSION = 1;
const PENDING_STORE = 'pending_runs';

function openRunDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(COLLECTOR_DB_NAME, COLLECTOR_DB_VERSION);
    req.onerror = () => reject(req.error);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(PENDING_STORE)) {
        db.createObjectStore(PENDING_STORE, { keyPath: 'runId' });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

function withStore(mode, fn) {
  return openRunDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(PENDING_STORE, mode);
        const store = tx.objectStore(PENDING_STORE);
        const request = fn(store);
        tx.oncomplete = () => resolve(request?.result ?? true);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      })
  );
}

function getAllPendingRuns() {
  return withStore('readonly', (store) => store.getAll());
}

function deletePendingRun(runId) {
  return withStore('readwrite', (store) => store.delete(runId));
}

async function markRunStatus(runId, status) {
  const db = await openRunDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(PENDING_STORE, 'readwrite');
    const store = tx.objectStore(PENDING_STORE);
    const getReq = store.get(runId);
    getReq.onerror = () => reject(getReq.error);
    getReq.onsuccess = () => {
      const run = getReq.result;
      if (run) {
        run.status = status;
        store.put(run);
      }
    };
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}

async function sendRun(run) {
  const body = {
    run_id: run.runId,
    points: run.points,
    meta: run.metadata?.meta || {},
    sensor_summary: run.metadata?.sensor_summary || null,
  };

  const res = await fetch('/api/collector/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`submit failed: ${text}`);
  }

  await deletePendingRun(run.runId);
}

async function processPendingRuns() {
  const runs = await getAllPendingRuns();
  const targets = runs.filter((r) => r.status === 'pending' || r.status === 'failed');
  for (const run of targets) {
    await markRunStatus(run.runId, 'sending');
    try {
      await sendRun(run);
    } catch (err) {
      console.warn('[sw] failed to send run', run.runId, err);
      await markRunStatus(run.runId, 'failed');
    }
  }

  const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
  clients.forEach((client) => {
    client.postMessage({ type: 'nodo-sync-updated' });
  });
}

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'nodo-sync-pending-runs') {
    event.waitUntil(processPendingRuns());
  }
});

self.addEventListener('message', (event) => {
  if (event.data === 'nodo-sync-now') {
    event.waitUntil(processPendingRuns());
  }
});
