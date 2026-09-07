import { HeadContent, Scripts, createRootRoute } from '@tanstack/react-router'
import DocsFooter from '../components/DocsFooter'
import DocsHeader from '../components/DocsHeader'
import { SITE } from '../lib/site'
import appCss from '../styles.css?url'

const THEME_INIT_SCRIPT = `(function(){try{var stored=window.localStorage.getItem('theme');var mode=(stored==='light'||stored==='dark'||stored==='auto')?stored:'auto';var prefersDark=window.matchMedia('(prefers-color-scheme: dark)').matches;var resolved=mode==='auto'?(prefersDark?'dark':'light'):mode;var root=document.documentElement;root.classList.remove('light','dark');root.classList.add(resolved);if(mode==='auto'){root.removeAttribute('data-theme')}else{root.setAttribute('data-theme',mode)}root.style.colorScheme=resolved;}catch(e){}})();`

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: `${SITE.title} · ${SITE.tagline}` },
      { name: 'description', content: SITE.description },
      { name: 'application-name', content: SITE.name },
      { name: 'theme-color', content: '#19191c' },
      { name: 'color-scheme', content: 'light dark' },
      { property: 'og:type', content: 'website' },
      { property: 'og:site_name', content: SITE.name },
      { property: 'og:title', content: `${SITE.name} · ${SITE.tagline}` },
      { property: 'og:description', content: SITE.description },
      { property: 'og:url', content: SITE.url },
      { property: 'og:image', content: `${SITE.url}/og.png` },
      { property: 'og:image:alt', content: `${SITE.name} documentation cover` },
      { name: 'twitter:card', content: 'summary_large_image' },
      { name: 'twitter:title', content: `${SITE.name} · ${SITE.tagline}` },
      { name: 'twitter:description', content: SITE.description },
      { name: 'twitter:image', content: `${SITE.url}/og.png` },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', href: '/favicon.svg', type: 'image/svg+xml' },
      { rel: 'manifest', href: '/site.webmanifest' },
      { rel: 'apple-touch-icon', href: '/favicon.svg' },
    ],
  }),
  shellComponent: RootDocument,
})

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
        <HeadContent />
      </head>
      <body>
        <div className="page-shell">
          <DocsHeader />
          {children}
          <DocsFooter />
        </div>
        <Scripts />
      </body>
    </html>
  )
}
