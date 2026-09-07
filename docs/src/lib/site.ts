export const SITE = {
  name: 'autogravity',
  title: 'autogravity docs',
  url: 'https://6a9e5e660013e2481b3d.appwrite.network',
  description:
    "Image focal-point detection as a microservice. Post an image, get back one coordinate pair: the weighted centre of the strongest salient region.",
  tagline: 'Find the subject. Crop nothing.',
  github: 'https://github.com/appwrite/autogravity',
  appwrite: 'https://appwrite.io',
  container: 'ghcr.io/appwrite/autogravity:latest',
  twitterHandle: '@appwrite',
} as const

export const NAV = {
  start: [
    { href: '#overview', label: 'Overview' },
    { href: '#preview', label: 'Storage preview' },
    { href: '#install', label: 'Install' },
    { href: '#config', label: 'Configuration' },
  ],
  reference: [
    { href: '#api', label: 'POST /analyze' },
    { href: '#limits', label: 'Limits' },
    { href: '#performance', label: 'Performance' },
  ],
} as const
