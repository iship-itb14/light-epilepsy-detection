/* Looping, silent videos: play only while on screen, never for reduced-motion
   users, and always pausable with the button inside the frame. */
(function(){
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('video[data-loop]').forEach(v => {
    const btn = v.parentElement.querySelector('.vctl');
    let userPaused = reduce;
    const sync = () => { if (btn) { btn.textContent = v.paused ? 'Play' : 'Pause'; btn.setAttribute('aria-pressed', String(v.paused)); } };
    v.addEventListener('play', sync); v.addEventListener('pause', sync);
    if (btn) btn.addEventListener('click', () => {
      if (v.paused) { userPaused = false; v.play().catch(() => {}); }
      else { userPaused = true; v.pause(); }
    });
    new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !userPaused) v.play().catch(() => {});
      else if (!e.isIntersecting) v.pause();
    }, { threshold: 0.35 }).observe(v);
    sync();
  });
})();
