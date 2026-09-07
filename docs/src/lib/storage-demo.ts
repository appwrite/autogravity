export const STORAGE_DEMO = {
  endpoint: 'https://fra.cloud.appwrite.io/v1',
  projectId: '6a9e5d640016fa6fb653',
  bucketId: '6a9ee9c60029f614d6cc',
  fileId: 'golden-retriever',
  cropSize: 400,
} as const

export function storagePreviewUrl(gravity: 'center' | 'auto') {
  const { endpoint, projectId, bucketId, fileId, cropSize } = STORAGE_DEMO
  const params = new URLSearchParams({
    project: projectId,
    width: String(cropSize),
    height: String(cropSize),
    gravity,
    quality: '90',
    output: 'jpg',
  })

  return `${endpoint}/storage/buckets/${bucketId}/files/${fileId}/preview?${params}`
}

export function storageViewUrl() {
  const { endpoint, projectId, bucketId, fileId } = STORAGE_DEMO
  return `${endpoint}/storage/buckets/${bucketId}/files/${fileId}/view?project=${projectId}`
}
