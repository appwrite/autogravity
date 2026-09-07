# autogravity docs

Documentation site for [autogravity](https://github.com/appwrite/autogravity), built with TanStack Start.

## Development

```sh
cd docs
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Build

```sh
npm run build
```

## Deploy

From the repository root:

```sh
appwrite client --endpoint "https://fra.cloud.appwrite.io/v1" --project-id "6a9e5d640016fa6fb653"
appwrite push sites --site-id autogravity-docs --force
```
