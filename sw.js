const K='spaceaudio-v2',F=['./','index.html','app.js','manifest.json'];
self.addEventListener('install',e=>e.waitUntil(caches.open(K).then(c=>c.addAll(F))));
self.addEventListener('fetch',e=>{if(new URL(e.request.url).origin!==location.origin)return;
e.respondWith(fetch(e.request).then(r=>{const c=r.clone();caches.open(K).then(x=>x.put(e.request,c));return r}).catch(()=>caches.match(e.request)))});
