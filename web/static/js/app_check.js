// web/static/js/app_check.js
document.addEventListener('DOMContentLoaded', () => {
  const fileBirth  = document.getElementById('fileBirth');
  const fileFather = document.getElementById('fileFather');
  const fileMother = document.getElementById('fileMother');

  const extractBirth  = document.getElementById('extractBirth');
  const extractFather = document.getElementById('extractFather');
  const extractMother = document.getElementById('extractMother');

  const previewImg  = document.getElementById('previewImg');
  const ocrText     = document.getElementById('ocrText');
  const paperTitle  = document.getElementById('paperTitle');

  const alertBox   = document.getElementById('alertBox');
  const alertText  = document.getElementById('alertText');
  const alertClose = document.getElementById('alertClose');

  const rightPane  = document.querySelector('.right');

  /* =========================
   *  QUY TẮC XÁC THỰC TIÊU ĐỀ
   * =========================
   * - mustGroups: mảng các nhóm; MỖI nhóm chỉ cần khớp 1 biến thể
   * - forbidAny : nếu xuất hiện bất kỳ cụm nào => sai
   * - canonical : tiêu đề chuẩn để tính điểm tham khảo (không quyết định pass/fail)
   */
  const DOC_RULES = {
    birth: {
      mustGroups: [
        ['dang ky', 'dang ki'],     // chấp nhận ký/ki
        ['khai sinh']
      ],
      forbidAny: ['khai tu', 'ket hon', 'giam ho', 'cap lai', 'chung nhan'],
      canonical: 'to khai dang ky khai sinh',
    },
    // Quy tắc kiểm CCCD (áp cho cả cha/mẹ)
    idcard: {
      mustGroups: [
        ['can cuoc cong dan', 'the can cuoc cong dan', 'can cuoc cong-dan', 'can cuoc cccd'],
      ],
      forbidAny: ['khai sinh', 'khai tu', 'ket hon', 'giam ho'],
      canonical: 'the can cuoc cong dan',
    },
  };

  /* ============ STATE / RACE ============ */
  const state = {
    birth:  { seq: 0, headCtrl: null, fullCtrl: null, busy: false },
    father: { seq: 0, headCtrl: null, fullCtrl: null, busy: false },
    mother: { seq: 0, headCtrl: null, fullCtrl: null, busy: false },
  };

  const VALID_TYPES = ['image/jpeg','image/png','image/webp','image/gif'];
  const isValid = f => !!f && VALID_TYPES.includes(f.type);

  function showAlert(ok, msg){
    alertText.textContent = msg;
    alertBox.classList.toggle('ok', !!ok);
    alertBox.classList.toggle('err', !ok);
    alertBox.style.display = 'flex';
  }
  alertClose?.addEventListener('click', ()=> alertBox.style.display='none');

  function setRightPanelImage(file){
    if (!file) return;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
  }

  /* ============ Chuẩn hoá & fuzzy ============ */
  function normalizeVN(str){
    return String(str||'')
      .normalize('NFD').replace(/[\u0300-\u036f]/g,'')
      .toLowerCase().replace(/[^a-z0-9\s-]/g,' ')   // giữ dấu '-' cho trường hợp "cong-dan"
      .replace(/\s+/g,' ').trim();
  }
  function levenshtein(a,b){
    a=a||''; b=b||'';
    const m=a.length, n=b.length;
    if(m===0) return n; if(n===0) return m;
    const dp=new Array(n+1).fill(0);
    for(let j=0;j<=n;j++) dp[j]=j;
    for(let i=1;i<=m;i++){
      let prev=dp[0]; dp[0]=i;
      for(let j=1;j<=n;j++){
        const tmp=dp[j];
        dp[j]=Math.min(dp[j]+1, dp[j-1]+1, prev + (a[i-1]===b[j-1]?0:1));
        prev=tmp;
      }
    }
    return dp[n];
  }
  function similarity(a,b){
    const A=normalizeVN(a), B=normalizeVN(b);
    if(!A || !B) return 0;
    const dist=levenshtein(A,B);
    return 1 - dist / Math.max(A.length,B.length);
  }

  // so khớp mềm 1 biến thể: chứa trực tiếp, hoán đổi y<->i, hoặc gần giống trên cửa sổ con
  function containsTokenSoft(H, token){
    const t = token.replace(/\s+/g,' ').trim();
    if (!t) return false;

    if (H.includes(t)) return true;

    // hoán đổi i<->y (ky/ki)
    const swap = t.replace(/y/g,'_Y_').replace(/i/g,'y').replace(/_Y_/g,'i');
    if (swap !== t && H.includes(swap)) return true;

    // so fuzzy trên cửa sổ con cùng số từ
    const hWords = H.split(' ');
    const tWords = t.split(' ');
    const k = tWords.length;
    if (hWords.length < k) return false;

    for (let i=0; i<=hWords.length-k; i++){
      const windowStr = hWords.slice(i, i+k).join(' ');
      if (similarity(windowStr, t) >= 0.85) return true;
    }
    return false;
  }

  function validateTitleByType(docType, headTitle){
    const rules = DOC_RULES[docType];
    const H = normalizeVN(headTitle);
    if (!rules || !H) return { ok:false, reason:'missing_rules_or_title', score:0 };

    // cấm
    if ((rules.forbidAny||[]).some(p => normalizeVN(p) && H.includes(normalizeVN(p)))){
      return { ok:false, reason:'forbidden_hit', score:0 };
    }

    // mỗi group phải khớp ít nhất 1 biến thể
    if (rules.mustGroups && rules.mustGroups.length){
      const allPass = rules.mustGroups.every(group =>
        group.some(variant => containsTokenSoft(H, normalizeVN(variant)))
      );
      if (!allPass) return { ok:false, reason:'missing_required_tokens', score:0 };
    }

    // điểm tham khảo
    const score = rules.canonical ? similarity(H, rules.canonical) : 1.0;
    return { ok:true, reason:'passed', score };
  }

  /* ============ Gọi OCR ============ */
  async function callOCR(file, {preset='fast', region='full', max_new_tokens=96, prompt='Chỉ trả về nội dung văn bản nhìn thấy trong ảnh.'}={}, abortSignal){
    const fd = new FormData();
    fd.append('file', file);
    fd.append('preset', preset);
    fd.append('region', region);
    fd.append('max_new_tokens', String(max_new_tokens));
    fd.append('prompt', prompt);
    const res = await fetch('/ocr', { method:'POST', body: fd, signal: abortSignal });
    return res.json();
  }

  async function getHeadTitle(file, abortSignal){
    const data = await callOCR(file, {
      preset: 'fast', region: 'head', max_new_tokens: 32,
      prompt: 'Chỉ trả JSON {"title":"..."} với tiêu đề chính của biểu mẫu. Không thêm chữ nào khác.'
    }, abortSignal);
    if (data?.success && typeof data.text === 'string'){
      const m = data.text.match(/"title"\s*:\s*"([^"]+)"/i);
      if (m) return m[1].trim();
      // fallback: lấy dòng có các từ khoá thường gặp
      const line = (data.text||'').split(/\r?\n/)
        .map(s=>s.trim())
        .find(s=>{
          const n = normalizeVN(s);
          return n.includes('to khai') || n.includes('dang ky') || n.includes('dang ki')
                 || n.includes('can cuoc') || n.includes('cccd') || n.includes('the can cuoc');
        });
      return (line || '').trim();
    }
    return '';
  }

  /* ============ Loading & disable ============ */
  function setBusy(which, busy){
    state[which].busy = busy;
    const mapBtn = { birth: extractBirth, father: extractFather, mother: extractMother };
    const btn = mapBtn[which];
    if (btn){
      btn.disabled = !!busy;
      btn.innerHTML = busy ? '<span class="spinner"></span> Đang trích xuất...' : 'Extract nội dung';
    }
    const anyBusy = state.birth.busy || state.father.busy || state.mother.busy;
    rightPane.classList.toggle('loading', anyBusy);
  }

  /* ============ Extract + hiển thị ============ */
  async function extractAndShow(which){
    const file = (which==='birth') ? fileBirth.files?.[0]
               : (which==='father') ? fileFather.files?.[0]
               : (which==='mother') ? fileMother.files?.[0]
               : null;

    if (!isValid(file)){
      showAlert(false, 'Vui lòng chọn ảnh hợp lệ.');
      return;
    }

    // tăng seq + huỷ request cũ
    state[which].seq += 1;
    const seqNow = state[which].seq;
    try { state[which].headCtrl?.abort(); } catch(e){}
    try { state[which].fullCtrl?.abort(); } catch(e){}
    state[which].headCtrl = new AbortController();
    state[which].fullCtrl = new AbortController();

    // reset UI
    setRightPanelImage(file);
    paperTitle.textContent = '—';
    ocrText.textContent = '(đang chờ)';
    alertBox.style.display = 'none';

    setBusy(which, true);

    // 1) Head title
    let headTitle = '';
    try{
      headTitle = await getHeadTitle(file, state[which].headCtrl.signal);
    }catch(err){
      if (err?.name === 'AbortError') return;
      showAlert(false, 'Lỗi OCR (Head).'); setBusy(which,false); return;
    }
    if (seqNow !== state[which].seq) return;

    paperTitle.textContent = headTitle || '—';

    // 2) Validate từng khu
    const DOC_TYPE = window.DOC_TYPE || 'birth';
    if (which === 'birth' && DOC_TYPE === 'birth'){
      const { ok } = validateTitleByType('birth', headTitle);
      showAlert(ok,
        ok ? 'Nội dung giấy tờ hợp lệ.'
           : 'Mẫu giấy tờ chưa hợp lệ, yêu cầu kiểm tra lại.'
      );
    }
    if (which === 'father'){
      const { ok } = validateTitleByType('idcard', headTitle);
      showAlert(ok,
        ok ? 'CCCD của cha hợp lệ.'
           : 'CCCD của cha chưa hợp lệ.'
      );
    }
    if (which === 'mother'){
      const { ok } = validateTitleByType('idcard', headTitle);
      showAlert(ok,
        ok ? 'CCCD của mẹ hợp lệ.'
           : 'CCCD của mẹ chưa hợp lệ.'
      );
    }

    // 3) OCR full
    let data;
    try{
      data = await callOCR(file, {
        preset: 'full', region: 'full', max_new_tokens: 256,
        prompt: 'Chỉ trả về nội dung văn bản nhìn thấy trong ảnh. Nếu là danh sách, xuống dòng theo từng mục.'
      }, state[which].fullCtrl.signal);
    }catch(err){
      if (err?.name === 'AbortError') return;
      showAlert(false, 'Lỗi OCR (Full).'); setBusy(which,false); return;
    }
    if (seqNow !== state[which].seq) return;

    if (data?.success){
      ocrText.textContent = data.text || '(trống)';
    } else {
      ocrText.textContent = 'Lỗi OCR: ' + (data?.error || 'unknown');
    }

    setBusy(which, false);
  }

  /* ============ Bindings ============ */
  extractBirth ?.addEventListener('click', ()=> extractAndShow('birth'));
  extractFather?.addEventListener('click', ()=> extractAndShow('father'));
  extractMother?.addEventListener('click', ()=> extractAndShow('mother'));

  [fileBirth, fileFather, fileMother].forEach(inp=>{
    inp?.addEventListener('change', (e)=>{
      const f = e.target.files?.[0];
      if (isValid(f)) setRightPanelImage(f);
    });
  });
});
