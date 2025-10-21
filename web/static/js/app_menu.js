// web/static/js/app_menu.js
document.addEventListener('DOMContentLoaded', ()=>{
  const docSelect = document.getElementById('docSelect');
  const goBtn = document.getElementById('goBtn');

  goBtn?.addEventListener('click', ()=>{
    const type = docSelect?.value || 'birth';
    window.location.href = `/check?type=${encodeURIComponent(type)}`;
  });
});
