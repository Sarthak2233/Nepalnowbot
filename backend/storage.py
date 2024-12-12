storage = {
    "dom_content": "",
    "refined_content": "",
    "anchor_tags": []
}

def save_to_storage(key, value):
    storage[key] = value

def get_from_storage(key):
    return storage.get(key, None)
