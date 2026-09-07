type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'

const METHOD_TONE: Record<HttpMethod, string> = {
  GET: 'endpoint-method--get',
  POST: 'endpoint-method--post',
  PUT: 'endpoint-method--put',
  PATCH: 'endpoint-method--patch',
  DELETE: 'endpoint-method--delete',
}

type HttpEndpointProps = {
  method: HttpMethod
  path: string
  description?: string
}

export default function HttpEndpoint({ method, path, description }: HttpEndpointProps) {
  return (
    <div className="endpoint-block">
      <div className="endpoint-row">
        <span className={`endpoint-method ${METHOD_TONE[method]}`}>{method}</span>
        <code className="endpoint-path">{path}</code>
      </div>
      {description ? <p className="endpoint-description">{description}</p> : null}
    </div>
  )
}
