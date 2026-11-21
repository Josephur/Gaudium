// Served at /sw.js while "/" is the Gaudium UI: clears any stale service worker cached from an older UI
// so the current Gaudium page always loads fresh. This replacement unregisters itself and reloads open pages.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    try { for (const k of await caches.keys()) await caches.delete(k); } catch (err) { /* no caches */ }
    await self.registration.unregister();
    for (const c of await self.clients.matchAll({ type: 'window' })) { try { c.navigate(c.url); } catch (err) { /* not navigable */ } }
  })());
});
