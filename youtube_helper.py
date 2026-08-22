from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def get_video_id(url):

    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in [
        "www.youtube.com",
        "youtube.com"
    ]:
        return parse_qs(
            parsed_url.query
        ).get("v", [None])[0]

    return None


def get_transcript_from_url(url):

    video_id = get_video_id(url)

    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    transcript = YouTubeTranscriptApi().fetch(video_id)

    full_transcript = " ".join(
        item.text for item in transcript
    )

    return full_transcript