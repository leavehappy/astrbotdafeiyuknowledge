import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from .learning import KnowledgeLearner
from .store import KnowledgeStore, public_query, public_url


class Clock:
    value = 1793750400.0

    def __call__(self):
        return self.value


class KnowledgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fish_knowledge_test_')
        self.clock = Clock()
        self.path = Path(self.temp.name) / 'knowledge.sqlite3'
        self.store = KnowledgeStore(self.path, now=self.clock)
        self.quote = '官方文档说明异步任务可以等待完成，也可以在满足条件后主动取消。'
        self.content = self.quote + ' 阅读文档时应同时记录适用版本和发布日期，以免把旧版本中的行为当成当前行为。' * 4
        self.calls = {'search': 0, 'fetch': 0}

        async def search(query):
            self.calls['search'] += 1
            return {'results': [{'url': 'https://docs.python.org/3/library/asyncio.html', 'title': '异步任务', 'date': '2026-11-04'},
                                {'url': 'https://reference.example.org/asyncio', 'title': '异步任务参考'}]}

        async def fetch(url):
            self.calls['fetch'] += 1
            return {'title': '异步任务', 'content': self.content}

        self.learner = KnowledgeLearner(self.store, {}, search, fetch)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def note(self, **kw):
        data = dict(title='Python异步任务', topic='Python异步任务', version='3.14',
                    summary='Python异步任务支持等待与取消。使用这些能力时应核对文档版本，并记录资料来源和核对日期。',
                    evidence=[{'url': 'https://docs.python.org/3/library/asyncio.html', 'quote': self.quote}], ttl_days=30)
        data.update(kw)
        return self.store.save(**data)

    def test_query_rejects_private_identifiers_and_keys(self):
        for query in ['查群友123456789住址', 'test@example.com是谁', 'Bearer abcdefghijk', 'sk-abcdefghijklmnop怎么用']:
            with self.assertRaises(ValueError):
                public_query(query)
        with self.assertRaises(ValueError):
            public_query('小明的私人偏好', ('小明',))
        self.assertEqual(public_query('Python 2026版本文档'), 'Python 2026版本文档')

    def test_urls_exclude_private_hosts(self):
        for url in ['file:///etc/passwd', 'http://127.0.0.1/', 'http://[::1]/', 'http://10.0.0.1/',
                    'http://localhost/', 'https://name:password@example.com/', 'https://x.local/']:
            with self.assertRaises(ValueError):
                public_url(url)
        self.assertEqual(public_url('https://example.org/doc?a=b&utm_source=x#top'), 'https://example.org/doc?a=b')

    def test_budget_is_atomic_and_persistent(self):
        self.assertTrue(self.store.reserve('search', 2))
        second = KnowledgeStore(self.path, now=self.clock)
        try:
            self.assertTrue(second.reserve('search', 2))
            self.assertFalse(self.store.reserve('search', 2))
            self.assertFalse(second.reserve('search', 2))
            self.assertEqual(self.store.stats()['usage']['search'], 2)
        finally:
            second.close()
        self.assertFalse(self.store.reserve('fetch', 0))

    def test_expired_knowledge_is_not_recalled(self):
        self.note(ttl_days=1)
        self.assertEqual(len(self.store.search('Python异步任务')), 1)
        self.clock.value += 86401
        self.assertEqual(self.store.search('Python异步任务'), [])
        self.assertEqual(self.store.stats()['expired'], 1)

    def test_duplicates_update_and_keep_revision(self):
        first = self.note()
        second = self.note(summary='Python异步任务支持等待完成和主动取消，最新文档已核对；仍需在实际使用时关注取消后的清理行为。')
        self.assertEqual(first['id'], second['id'])
        self.assertTrue(second['updated_existing'])
        self.assertEqual(self.store.stats()['total'], 1)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 1)

    def test_versions_stay_separate(self):
        self.note()
        newer = self.note(version='3.15', summary='Python新版本异步任务的行为应以当前版本文档为准，跨版本使用时需要核对新增接口和取消语义。')
        self.assertFalse(newer['updated_existing'])
        self.assertEqual(self.store.stats()['total'], 2)

    def test_identical_note_does_not_create_revision_noise(self):
        first = self.note()
        again = self.note()
        self.assertEqual(first['id'], again['id'])
        self.assertTrue(again['duplicate_skipped'])
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM revisions').fetchone()[0], 0)
        newer = self.note(version='3.15')
        self.assertNotEqual(first['id'], newer['id'])

    def test_unrelated_content_is_not_injected(self):
        self.note()
        self.assertEqual(self.store.search('今天晚上吃什么火锅'), [])

    def test_feedback_requires_user_quote_and_is_scoped(self):
        with self.assertRaises(ValueError):
            self.store.save_feedback('a', 'questions', '少反问', '你别反问了', '我今天吃饭了')
        self.store.save_feedback('a', 'questions', '减少无必要的追问', '你别总是反问', '你别总是反问')
        self.assertEqual(self.store.feedback('a'), ['减少无必要的追问'])
        self.assertEqual(self.store.feedback('b'), [])

    async def test_search_cache_prevents_repeat_billing(self):
        first = await self.learner.learn('Python异步任务', 'chat')
        again = await self.learner.learn('Python异步任务', 'chat')
        self.assertEqual(first['receipt_id'], again['receipt_id'])
        self.assertEqual(self.calls, {'search': 1, 'fetch': 2})

    async def test_save_requires_real_quote_and_matching_session(self):
        pack = await self.learner.learn('Python异步任务', 'chat')
        evidence = json.dumps([{'url': pack['sources'][0]['url'], 'quote': self.quote}], ensure_ascii=False)
        with self.assertRaises(ValueError):
            await self.learner.save(pack['receipt_id'], 'another_chat', '异步任务', self.content[:90], evidence)
        forged = json.dumps([{'url': pack['sources'][0]['url'], 'quote': '这段文字并不存在于网页正文中，不能作为已经查证的证据。'}], ensure_ascii=False)
        with self.assertRaises(ValueError):
            await self.learner.save(pack['receipt_id'], 'chat', '异步任务', self.content[:90], forged)
        result = await self.learner.save(pack['receipt_id'], 'chat', '异步任务', self.content[:90], evidence, '3.14', 'current')
        self.assertEqual(result['expires'], self.clock.value + 86400)
        self.assertEqual(self.store.stats()['active'], 1)

    async def test_untrusted_single_domain_is_not_enough(self):
        pack = await self.learner.learn('Python异步任务', 'chat')
        page = pack['sources'][1]
        evidence = json.dumps([{'url': page['url'], 'quote': self.quote}], ensure_ascii=False)
        with self.assertRaises(ValueError):
            await self.learner.save(pack['receipt_id'], 'chat', '异步任务', self.content[:90], evidence)
        evidence = json.dumps([{'url': p['url'], 'quote': self.quote} for p in pack['sources']], ensure_ascii=False)
        self.assertIn('id', await self.learner.save(pack['receipt_id'], 'chat', '异步任务', self.content[:90], evidence))

    async def test_failed_fetch_attempts_are_also_bounded(self):
        async def fail(url):
            self.calls['fetch'] += 1
            raise RuntimeError('offline')
        self.learner.fetch_api = fail
        with self.assertRaises(ValueError):
            await self.learner.learn('Python异步任务', 'chat')
        self.assertEqual(self.calls['fetch'], 2)
        self.assertEqual(self.store.stats()['total'], 0)

    async def test_embedding_failure_falls_back_to_keywords(self):
        async def fail(text):
            raise RuntimeError('embedding offline')
        self.note()
        self.learner.embed = fail
        found = await self.learner.lookup('Python异步任务的取消')
        self.assertEqual(len(found), 1)

    async def test_private_search_never_reaches_network(self):
        with self.assertRaises(ValueError):
            await self.learner.learn('QQ 123456789是谁', 'chat')
        self.assertEqual(self.calls['search'], 0)


if __name__ == '__main__':
    unittest.main()
