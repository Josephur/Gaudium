import { html, render } from './ui/h.js';
import { App, boot } from './app.js';

render(html`<${App} />`, document.getElementById('app'));
boot();
