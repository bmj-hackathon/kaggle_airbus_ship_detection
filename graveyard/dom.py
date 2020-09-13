#%%
class DOM_test:

    def __init(self):
        pass

    def P(self, **kwargs):
        self.make_element('P', **kwargs)

    def make_element(self, type):
        this_el = getattr(html,type)
        return this_el


# dom = DOM()
# r = dom.P()