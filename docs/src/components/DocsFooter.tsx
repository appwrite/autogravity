import AppwriteLogo from './AppwriteLogo'
import { SITE } from '../lib/site'

export default function DocsFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer-meta">
        <span>{SITE.container}</span>
        <a href={SITE.github}>{SITE.github.replace('https://github.com/', '')}</a>
      </div>

      <a
        href={SITE.appwrite}
        target="_blank"
        rel="noreferrer"
        className="site-footer-credit"
        aria-label="Built by Appwrite"
      >
        <span className="site-footer-credit-label">Built by</span>
        <AppwriteLogo className="site-footer-logo" />
        <span className="site-footer-credit-name">Appwrite</span>
      </a>
    </footer>
  )
}
