import redditwarp.SYNC
from config import CONFIG
from redditwarp.models.submission import LinkPost, TextPost, GalleryPost, Submission
from typing import Optional, Sequence

class RedditToolset:
    def __init__(self):
        self.client = redditwarp.SYNC.Client(
            CONFIG["app_id"],
            CONFIG["secret"],
            CONFIG["refresh_token"],
        )

    def search(self, query: str, subreddit: str = "", limit: int = 25):
        result = self.client.p.submission.search(
            sr=subreddit,
            query=query,
            sort="relevance",
            amount=10,
            time="all"
        )
        return result

    def fetch_hot_posts(self, subreddit: str, limit: int = 10):
        return self.client.p.subreddit.pull.hot(subreddit, limit)

    def fetch_post_with_comments(self, post_id: str, comment_limit: int = 1000, comment_depth: int = 5):
        submission = self.client.p.submission.fetch(post_id)
        content = (
            f"Title: {submission.title}\n"
            f"Score: {submission.score}\n"
            f"Author: {submission.author_display_name or '[deleted]'}\n"
            f"Type: {_get_post_type(submission)}\n"
            f"Content: {_get_content(submission)}\n"
        )

        comments = self.client.p.comment_tree.fetch(post_id, sort='top', limit=comment_limit, depth=comment_depth)
        if comments.children:
            content += "\nComments:\n"
            for comment in comments.children:
                content += "\n" + _format_comment_tree(comment)
        else:
            content += "\nNo comments found."
        return content


def _format_comment_tree(comment_node, depth: int = 0) -> str:
    """Helper method to recursively format comment tree with proper indentation"""
    comment = comment_node.value
    indent = "-- " * depth
    content = (
        f"{indent}* Author: {comment.author_display_name or '[deleted]'}\n"
        f"{indent}  Score: {comment.score}\n"
        f"{indent}  {comment.body}\n"
    )

    for child in comment_node.children:
        content += "\n" + _format_comment_tree(child, depth + 1)

    return content


def _get_post_type(submission) -> str:
    """Helper method to determine post type"""
    if isinstance(submission, LinkPost):
        return 'link'
    elif isinstance(submission, TextPost):
        return 'text'
    elif isinstance(submission, GalleryPost):
        return 'gallery'
    return 'unknown'

def _get_content(submission) -> Optional[str]:
    """Helper method to extract post content based on type"""
    if isinstance(submission, LinkPost):
        return submission.permalink
    elif isinstance(submission, TextPost):
        return submission.body
    elif isinstance(submission, GalleryPost):
        return str(submission.gallery_link)
    return None
