// Usage (with the dashboard running):
//   node tools/check_page.js http://<host>:5000 templates/index.html
//
// Run the dashboard page's script against the live server with a stand-in DOM,
// so a runtime error in a render function shows up without opening a browser.
const fs = require('fs');
const BASE = process.argv[2];
const html = fs.readFileSync(process.argv[3], 'utf8');
const script = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));

const elements = new Map();
function element(key) {
  if (elements.has(key)) return elements.get(key);
  const listeners = {};
  const el = {
    key, innerHTML: '', textContent: '', hidden: false, value: '', dataset: {}, children: [],
    style: {setProperty() {}}, scrollTop: 0, checked: false,
    classList: {_s: new Set(), add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
                toggle(c, on) { on ? this._s.add(c) : this._s.delete(c); }, contains(c) { return this._s.has(c); }},
    setAttribute(k, v) { this[`attr_${k}`] = v; }, getAttribute(k) { return this[`attr_${k}`]; },
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    fire(type, event) { (listeners[type] || []).forEach(fn => fn(event || {target: el})); },
    appendChild(child) { this.children.push(child); return child; },
    scrollIntoView() {}, querySelectorAll() { return []; }, querySelector() { return element(key + ' *'); },
  };
  elements.set(key, el);
  return el;
}
global.document = {
  querySelector: sel => element(sel),
  querySelectorAll: () => [],
  createElement: tag => element(`<${tag}>#${Math.random()}`),
  createDocumentFragment: () => element(`<fragment>#${Math.random()}`),
  body: element('body'),
};
global.matchMedia = () => ({matches: true});
global.history = {replaceState() {}};
global.location = {hash: ''};
const realFetch = global.fetch;
global.fetch = url => realFetch(BASE + url);

const failures = [];
process.on('unhandledRejection', e => failures.push('unhandled: ' + (e && e.stack || e)));

(async () => {
  // Expose the page's functions and state; the script is otherwise unchanged.
  const exposed = new Function(script + `
    return {state, simState, rank, loadMeta, setView, setSimSub, selectSimGame, setSimTeam, loadRank,
            renderSimStandings, loadSimLeaders};`)();
  const settle = ms => new Promise(r => setTimeout(r, ms));
  await settle(4000);                                   // the page's own start-up: meta + games
  const {state, simState, rank} = exposed;
  const shown = sel => element(sel).innerHTML;
  const check = (name, ok, detail) => {
    console.log(`${ok ? 'OK ' : 'HATA'}  ${name}${detail ? ' - ' + detail : ''}`);
    if (!ok) failures.push(name);
  };
  check('meta yuklendi, sim sezonu biliniyor', state.meta && state.meta.sim_season === '2026_2027', state.meta && state.meta.sim_season);
  check('Sim sekmesi gorunur', element('#simTab').hidden === false);

  exposed.setSimSub('standings'); await settle(2500);
  check('puan durumu tablosu', shown('#simStandTable').includes('<table class="stand">') && (shown('#simStandTable').match(/<tr class=/g) || []).length === 30,
        `${(shown('#simStandTable').match(/<tr class=/g) || []).length} satir`);
  check('uyari bandi', shown('#simNote').includes('gerçek sonuç değil'));
  simState.conf = 'east'; exposed.renderSimStandings();
  check('Dogu konferansi', (shown('#simStandTable').match(/<tr class=/g) || []).length === 15 && shown('#simStandTable').includes(' po"'));
  simState.conf = 'all';

  exposed.setSimSub('leaders'); await settle(2500);
  const cards = (shown('#simLeadersGrid').match(/class="card lcard"/g) || []).length;
  check('lig liderleri kartlari', cards === 12, `${cards} kategori`);
  check('lider satiri ve yuzde bicimi', shown('#simLeadersGrid').includes('class="first"') && /\d+,\d%/.test(shown('#simLeadersGrid')));
  exposed.setSimTeam('OKC'); await settle(2500);
  check('takim liderleri (OKC)', shown('#simLeadersHead').includes('takım liderleri') && shown('#simLeadersGrid').includes('Gilgeous-Alexander'));
  exposed.setSimTeam(''); await settle(1500);

  exposed.setSimSub('games'); await settle(2500);
  check('simule mac listesi', element('#simGameCount').textContent.includes('1200 simüle maç'), element('#simGameCount').textContent);
  await exposed.selectSimGame('0022600003'); await settle(500);
  const detail = shown('#simDetail');
  check('mac detayi: skor, kutu skoru, play-by-play', detail.includes('simüle final') && (detail.match(/class="data boxt"/g) || []).length === 2 && detail.includes('id="simPlays"'));
  check('play-by-play satirlari (1. ceyrek)', (shown('#simPlays').match(/class="play/g) || []).length > 60, `${(shown('#simPlays').match(/class="play/g) || []).length} oyun`);

  exposed.setSimSub('rank'); await settle(3500);
  check('sim guc siralamasi', rank.sim === true && (shown('#rankTable').match(/class="rrow/g) || []).length === 30 && shown('#rankMeta').includes('SİM'));
  check('sim oyuncu OVR listesi, uretim sutunu yok', shown('#rankPlayers').includes('plrank') && !shown('#rankPlayers').includes('üretim'));
  exposed.rank.window = 'week'; exposed.rank.key = null; await exposed.loadRank(); await settle(500);
  check('sim haftalik siralama', (shown('#rankTable').match(/class="rrow/g) || []).length === 30, shown('#rankMeta').replace(/<[^>]+>/g, '').slice(0, 60));

  exposed.setView('rank'); await settle(3500);
  check('gercek guc siralamasina donus', rank.sim === false && shown('#rankPlayers').includes('üretim') && !shown('#rankMeta').includes('SİM'));
  exposed.setView('games'); await settle(500);
  check('gercek maclar gorunumu', element('#simBar').hidden === true && element('#main').hidden === false);

  console.log(failures.length ? `\n${failures.length} HATA:\n` + failures.join('\n') : '\nhepsi gecti');
  process.exit(failures.length ? 1 : 0);
})().catch(e => { console.error('CALISMA HATASI', e.stack || e); process.exit(2); });
