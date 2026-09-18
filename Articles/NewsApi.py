from newsapi import NewsApiClient
from datetime import datetime, timedelta
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from Vars import Category
except ImportError:
    from Articles.Vars import Category



class NewsAPI():
    def __init__(self, apiKey):
        self.client = NewsApiClient(api_key = apiKey)
        self._sources = self._getsource()


    def _getsource(self):
        '''function for returning all available sources for news'''
        return self.client.get_sources(category = Category)


    def _source(self):
        '''alias for _getsource'''
        return self._getsource()


    def _NewsQueryByKeyword(self, topic: str) -> list:
        '''news query function by just keyword (q) searching all domains, from past week sorted by publishedAt'''
        self._getsource()
        try:
            one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            queries1 = self.client.get_everything(
                q = topic,
                from_param = one_week_ago,
                sort_by = 'publishedAt'
            )
            queries2 = self.client.get_top_headlines(
                q = topic
            )
            return [queries1, queries2]
        except Exception as e:
            print(f"error news by keyword: {e}")
            return None


    def _NewsQueryByCategory(self, topic: str = None, category: str = Category) -> list:
        '''news query function searching top headlines by category'''
        self._getsource()
        try:
            queries1 = self.client.get_top_headlines(
                q = topic,
                category = category
            )
            queries2 = self.client.get_top_headlines(
                q = topic
            )
            return [queries1, queries2]
        except Exception as e:
            print(f"error news by category: {e}")
            return None


    def _NewsQueryBySource(self, source: str, topic: str = None) -> list:
        '''news query function searching by specific source(s), from past week sorted by publishedAt'''
        self._getsource()
        try:
            one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            queries1 = self.client.get_everything(
                q = topic,
                sources = source,
                from_param = one_week_ago,
                sort_by = 'publishedAt'
            )
            queries2 = self.client.get_everything(
                q = topic,
                from_param = one_week_ago,
                sort_by = 'publishedAt'
            )
            return [queries1, queries2]
        except Exception as e:
            print(f"error news by source: {e}")
            return None


    def _NewsQueryEverything(self, source: str = None, topic: str = None) -> list:
        '''news query function returns everything from past week sorted by publishedAt, with optional source or topic'''
        self._getsource()
        try:
            one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            kwargs = {
                'q': topic,
                'from_param': one_week_ago,
                'sort_by': 'publishedAt'
            }
            if source:
                kwargs['sources'] = source
            queries1 = self.client.get_everything(**kwargs)
            queries2 = self.client.get_everything(
                q = topic,
                from_param = one_week_ago,
                sort_by = 'publishedAt'
            )
            return [queries1, queries2]
        except Exception as e:
            print(f"error news everything: {e}")
            return None


    def _NewsQueryHeadlines(self, source: str = None, topic: str = None) -> list:
        '''news query headlines by source or by category'''
        self._getsource()
        try:
            if source:
                queries1 = self.client.get_top_headlines(q = topic, sources = source)
            else:
                queries1 = self.client.get_top_headlines(q = topic)
            queries2 = self.client.get_top_headlines(q = topic)
            return [queries1, queries2]
        except Exception as e:
            print(f"error news top headlines: {e}")
            return None
     


        

        

        
        

    