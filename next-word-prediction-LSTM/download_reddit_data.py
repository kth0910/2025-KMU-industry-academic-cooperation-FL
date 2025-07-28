from psaw import PushshiftAPI
import json

api = PushshiftAPI()
gen = api.search_comments(limit=100000, subreddit='all')
with open('reddit.jsonl', 'w') as f:
    for comment in gen:
        # author, body 포함
        f.write(json.dumps({
            'author': comment.author,
            'body': comment.body
        }) + '\\n')