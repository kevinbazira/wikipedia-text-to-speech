from a2wsgi import ASGIMiddleware

from wiki_tts.routes import app as fastapi_app

# Wrap our modern FastAPI app so Toolforge's uWSGI server can run it
app = ASGIMiddleware(fastapi_app)
