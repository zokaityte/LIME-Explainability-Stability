import uuid

def generate_short_uuid():
    return str(uuid.uuid4())[:8]
