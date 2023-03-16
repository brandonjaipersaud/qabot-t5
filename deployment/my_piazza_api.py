from collections import namedtuple
from piazza_api import network, rpc


class MyNetwork(network.Network):
    def __init__(self, network_id, session):
        self._nid = network_id
        self._rpc = rpc.PiazzaRPC(network_id=self._nid)
        self._rpc.session = session

        ff = namedtuple('FeedFilters', ['unread', 'following', 'folder'])
        self._feed_filters = ff(network.UnreadFilter, network.FollowingFilter, network.FolderFilter)

    
    def iter_all_posts(self, piazzabot_id, limit=None):
        """Get all posts visible to the current user

        This grabs you current feed and ids of all posts from it; each post
        is then individually fetched. This method does not go against
        a bulk endpoint; it retrieves each post individually, so a
        caution to the user when using this.

        :type limit: int|None
        :param limit: If given, will limit the number of posts to fetch
            before the generator is exhausted and raises StopIteration.
            No special consideration is given to `0`; provide `None` to
            retrieve all posts.
        :returns: An iterator which yields all posts which the current user
            can view
        :rtype: generator
        """
        feed = self.get_feed(limit=999999, offset=0)
        # returns feed array of json
        feed = self._filter_feed(feed, piazzabot_id)
        cids = [post['id'] for post in feed]
        if limit is not None:
            num_filtered = min(limit, len(feed))
            cids = cids[:num_filtered]
        for cid in cids:
            yield self.get_post(cid)


    def create_followup(self, post, content, anonymous=False):
        """Create a follow-up on a post `post`.

        It seems like if the post has `<p>` tags, then it's treated as HTML,
        but is treated as text otherwise. You'll want to provide `content`
        accordingly.

        :type  post: dict|str|int
        :param post: Either the post dict returned by another API method, or
            the `cid` field of that post.
        :type  content: str
        :param content: The content of the followup.
        :type  anonymous: bool
        :param anonymous: Whether or not to post anonymously.
        :rtype: dict
        :returns: Dictionary with information about the created follow-up.
        """
        try:
            cid = post["id"]
        except KeyError:
            cid = post

        params = {
            "cid": cid,
            "type": "followup",

            # For followups, the content is actually put into the subject.
            "subject": content,
            "content": "",
            "config": {
                "ionly": "true"
            },
            "anonymous": "yes" if anonymous else "no",
        }
        return self._rpc.content_create(params)

    """
    only retrieve questions with no answer
    use same filtering criteria in 2nd pass in run_bot.py
    Ways to detect if a question is unanswered:
        1. p['type'] == 'question' and 'unanswered' in p['tags']
        2. no_answer:1
        3. has_s = 1 or has_i = 1

        ** does piazzabot have a unique id?? avoid calling get_users()
    
    do piazzabot response check later due to throttling
    there shouldn't be any throttling here
    """
    def _filter_feed(self, feed: dict, piazzabot_id):
        filtered_feed = []
        # print(self.get_all_users())
        # print(feed.keys())
        # contains all 3337 Piazza posts
        feed = feed['feed']

        print(f'feed length before filtering {len(feed)}') 
        for p in feed:
            # if p['type'] == 'question':
            
            piazzabot_responded = False
            # unanswered questions that don't contain piazzabot in followup
            # add 'student' in p['tags'] for student question filtering

            # **notes that have been convered to questions are not marked as unanswered**
            if p['type'] == 'question' and p['no_answer']:
                # print(p)
                # check for piazzabot followups 
                for action in p['log']:
                    # user = self.get_users([action['u']])[0]['name']
                    # print(f'User is {user}')
                    # can also run a linear search to find piazzabot id from class instance
                    if action['n'] == 'followup' and action['u'] == piazzabot_id:
                        # skip adding this post since bot already responded
                        piazzabot_responded = True
                        break 
                if not piazzabot_responded:
                    filtered_feed.append(p)

        # print(filtered_feed)

        # for p in filtered_feed:
        #     print("-" * 50)
        #     print(f'Subject: {p["subject"]}')
        #     print(f'Content Snippet: {p["content_snipet"]}')
        #     print("-" * 50)
        # print(filtered_feed[0])
        print(f'feed length after filtering {len(filtered_feed)}') 

        return filtered_feed 