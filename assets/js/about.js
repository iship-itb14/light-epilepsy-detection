/* About us: click a portrait to open that person's story.
   The other two portraits shrink into a row on top and stay clickable.
   Movement uses FLIP (measure, change layout, animate the difference)
   so photos glide to their new place instead of jumping. */
(function () {
  var crew = document.querySelector('.crew');
  if (!crew) return;
  var mates = Array.prototype.slice.call(crew.querySelectorAll('.mate'));
  var still = window.matchMedia('(prefers-reduced-motion: reduce)');
  var DUR = 560;
  var EASE = 'cubic-bezier(.2,.7,.2,1)';

  function rects() {
    return mates.map(function (m) { return m.querySelector('.shot').getBoundingClientRect(); });
  }

  function set(id) {
    var before = rects();
    var current = crew.getAttribute('data-open') || '';
    var next = (id === current) ? '' : id;

    crew.setAttribute('data-open', next);
    crew.classList.toggle('open', !!next);
    mates.forEach(function (m) {
      var on = m.id === next;
      m.classList.toggle('on', on);
      m.querySelector('.shot').setAttribute('aria-expanded', on ? 'true' : 'false');
      m.querySelector('.more').hidden = !on;
    });

    if (still.matches) return;

    var after = rects();
    mates.forEach(function (m, i) {
      var a = before[i], b = after[i];
      if (!b.width) return;
      var dx = a.left - b.left, dy = a.top - b.top, s = a.width / b.width;
      m.querySelector('.shot').animate(
        [{ transform: 'translate(' + dx + 'px,' + dy + 'px) scale(' + s + ')' }, { transform: 'none' }],
        { duration: DUR, easing: EASE }
      );
      var meta = m.querySelectorAll('.meta, .more');
      meta.forEach(function (el) {
        if (el.hidden) return;
        el.animate([{ opacity: 0, transform: 'translateY(16px)' }, { opacity: 1, transform: 'none' }],
          { duration: 420, delay: 160, easing: EASE, fill: 'backwards' });
      });
    });
  }

  mates.forEach(function (m) {
    m.querySelector('.shot').addEventListener('click', function () {
      var wasOpen = crew.classList.contains('open');
      set(m.id);
      if (crew.classList.contains('open') && !wasOpen) {
        crew.scrollIntoView({ behavior: still.matches ? 'auto' : 'smooth', block: 'start' });
      }
    });
  });
  var back = crew.parentNode.querySelector('.crew-back');
  if (back) back.addEventListener('click', function () { set(crew.getAttribute('data-open')); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && crew.classList.contains('open')) set(crew.getAttribute('data-open'));
  });
})();
