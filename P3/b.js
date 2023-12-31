(window.webpackJsonp = window.webpackJsonp || []).push([
    [58, 66], {
        473: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return a
            }));
            const a = {
                AUTHOR_PAPER: "authorPaper",
                PAPER_CITATION: "paperCitation",
                PAPER_REFERENCE: "paperReference"
            }
        },
        474: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return c
            }));
            var a = r(9),
                n = r(15),
                i = r(8),
                s = r.n(i);

            function o(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function l(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class c extends n.PureComponent {
                getChildContext() {
                    const {
                        heapPropsChain: e = {}
                    } = this.context, {
                        heapProps: t = {}
                    } = this.props;
                    return {
                        heapPropsChain: "function" == typeof t ? t(e) : (function(e, t) {
                            if (p) return;
                            const r = function(e, t) {
                                return Object.keys(e).filter(e => e in t).filter(r => e[r] !== t[r])
                            }(e, t);
                            r.length > 0 && (a.a.warn(`<HeapTracking/> props overwritten (see: ${r.join(", ")})`), p = !0)
                        }(e, t), function(e) {
                            for (var t = 1; t < arguments.length; t++) {
                                var r = null != arguments[t] ? arguments[t] : {};
                                t % 2 ? o(Object(r), !0).forEach((function(t) {
                                    l(e, t, r[t])
                                })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : o(Object(r)).forEach((function(t) {
                                    Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                                }))
                            }
                            return e
                        }({}, e, {}, t))
                    }
                }
                render() {
                    return this.props.children || null
                }
            }
            l(c, "childContextTypes", {
                heapPropsChain: s.a.object.isRequired
            }), l(c, "contextTypes", {
                heapPropsChain: s.a.object
            });
            let p = !1
        },
        478: function(e, t, r) {
            "use strict";
            r.r(t);
            var a = r(380),
                n = r(525),
                i = r(382),
                s = r(375),
                o = r(43),
                l = r(22),
                c = r(0),
                p = r(15),
                u = r(368),
                d = r.n(u);

            function h() {
                return (h = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }
            const m = c.b.List([l.b.RESULT, l.b.METHODOLOGY, l.b.BACKGROUND, null]);
            class b extends p.PureComponent {
                renderReferencesExcerptIntentTitle(e) {
                    return Object(o.c)(t => t.references.excerptIntentLabel[e])
                }
                renderCitationsExcerptIntentTitle(e) {
                    const {
                        citedPaperTitle: t
                    } = this.props;
                    return p.createElement(p.Fragment, null, Object(o.c)(t => t.citations.excerptIntentLabel[e]), '"', p.createElement(i.c, {
                        field: t,
                        limit: 50,
                        expandable: !1
                    }), '"')
                }
                render() {
                    const {
                        citationType: e,
                        citationContexts: t,
                        className: r,
                        heapProps: a
                    } = this.props, u = function(e) {
                        const t = (e, t, r) => {
                            const a = e.get(t) || c.b.List();
                            return e.set(t, a.push(r))
                        };
                        return e.reduce((e, r) => r.intents.length > 0 ? r.intents.reduce((e, a) => t(e, a, r.context), e) : t(e, null, r.context), c.b.Map())
                    }(t).sortBy((e, t) => m.indexOf(t));
                    return p.createElement("div", h({
                        className: d()("cl-citation__excerpts", r)
                    }, a ? Object(s.a)(a) : {}), [...u.keys()].map(t => p.createElement(n.a, {
                        key: t || "null",
                        title: t ? e === l.a.CITED_PAPERS ? this.renderReferencesExcerptIntentTitle(t) : this.renderCitationsExcerptIntentTitle(t) : Object(o.c)(e => e.citations.excerptIntentLabel.none)
                    }, u.get(t).take(10).map((e, r) => p.createElement("div", {
                        key: `${t||"null"}.${r}`,
                        className: "cl-citation__excerpts__citation"
                    }, p.createElement(i.c, {
                        field: e
                    }))))))
                }
            }
            var f = r(587),
                g = r(397),
                y = r(448),
                O = r(409),
                E = r(182),
                v = r(10),
                P = r(61),
                S = r(89),
                w = r(130),
                _ = r(13),
                C = r(8),
                x = r.n(C);

            function j() {
                return (j = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function T(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            r.d(t, "INTENT", (function() {
                return k
            })), r.d(t, "EXCERPT", (function() {
                return I
            })), r.d(t, "HIGHLY_INFLUENCED", (function() {
                return N
            })), r.d(t, "TRUNCATE_TITLE_LENGTH", (function() {
                return D
            })), r.d(t, "default", (function() {
                return R
            })), r.d(t, "exampleConfig", (function() {
                return A
            }));
            const k = "intent",
                I = "excerpt",
                N = "highlyInfluenced",
                D = 50;
            class R extends p.PureComponent {
                constructor() {
                    super(...arguments), T(this, "_onClickFlagAnalytics", null), T(this, "_setControlsExpandedContent", null), T(this, "setAnalyticsCallbacks", e => {
                        let {
                            onClickFlag: t
                        } = e;
                        this._onClickFlagAnalytics = t
                    }), T(this, "onClickExcerptAndIntents", () => {
                        this.onClickBox(I)
                    }), T(this, "clearBoxes", () => {
                        this.setState({
                            shownBoxType: null
                        })
                    }), this.state = {
                        shownBoxType: null
                    }
                }
                onClickBox(e) {
                    this._onClickFlagAnalytics && this._onClickFlagAnalytics({
                        citationType: e
                    });
                    const {
                        onClick: t
                    } = this.props;
                    "function" == typeof t && t();
                    const {
                        shownBoxType: r
                    } = this.state;
                    if (this._setControlsExpandedContent) {
                        const t = e === r ? null : this.renderExcerptContext();
                        this._setControlsExpandedContent(t)
                    }
                    this.setState({
                        shownBoxType: e === r ? null : e
                    })
                }
                getIntentsLabel() {
                    const {
                        shouldRenderIntents: e,
                        citation: t,
                        citationType: r
                    } = this.props, {
                        envInfo: a
                    } = this.context;
                    if (e && !a.isMobile) {
                        const e = t.citationContexts.reduce((e, t) => t.intents.reduce((e, t) => e.add(t), e), c.b.Set());
                        if (e.size > 0) {
                            const t = e.map(e => Object(o.c)(t => t.citations.intentLabels[e])),
                                a = Object(O.c)([...t]);
                            return Object(o.c)(e => e.citations.intentTypeLabel[r], a)
                        }
                    }
                }
                getExcerptsLabel() {
                    const {
                        citation: e
                    } = this.props, t = e.citationContexts.size;
                    if (t > 0) return Object(P.j)(t, Object(o.c)(e => e.citations.excerpts.singular, t), Object(o.c)(e => e.citations.excerpts.plural, t), !0)
                }
                getExcerptAndIntentsFlag() {
                    const {
                        citation: e,
                        citationType: t
                    } = this.props, {
                        shownBoxType: r
                    } = this.state, a = this.getIntentsLabel(), n = this.getExcerptsLabel();
                    if (!a && !n) return null;
                    const i = [n, a].filter(Boolean).join(", ").toLowerCase(),
                        s = Object(O.a)(i),
                        o = {
                            "aria-expanded": r === I || r === k
                        };
                    return p.createElement(f.a, {
                        ariaProps: o,
                        onClick: this.onClickExcerptAndIntents,
                        label: s,
                        heapProps: {
                            id: E.e,
                            "paper-id": e.id,
                            type: I,
                            "citation-type": t
                        }
                    })
                }
                renderFlags() {
                    const {
                        excerpts: e
                    } = this.props, {
                        shownBoxType: t
                    } = this.state, {
                        envInfo: r
                    } = this.context, a = !!e && (!0 === e ? this.getExcerptAndIntentsFlag() : e);
                    if (!a) return null;
                    const n = t === I || t === k;
                    return r.isMobile ? p.createElement("div", {
                        className: "cl-paper-flags__paper-card"
                    }, a && p.createElement("div", {
                        className: "cl-paper-flags__paper-card__flag"
                    }, a)) : p.createElement("ul", {
                        className: "cl-paper__bulleted-row"
                    }, a && p.createElement("li", {
                        className: "cl-paper__bulleted-row__item cl-paper-flag__popover-button"
                    }, a && p.createElement(p.Fragment, null, a, n && p.createElement("div", {
                        className: "cl-paper-flag__popover__arrow"
                    }))))
                }
                renderBoxContext(e) {
                    if (!e) return null;
                    switch (e) {
                        case I:
                            return this.renderExcerptContext()
                    }
                    return Object(v.default)("paper-flags", `render method for ${e} is missing`), null
                }
                renderExcerptContext() {
                    const {
                        citation: e,
                        citationType: t,
                        citedPaperTitle: r
                    } = this.props;
                    return p.createElement(b, {
                        citationType: t,
                        citationContexts: e.citationContexts,
                        citedPaperTitle: r,
                        className: "cl-paper-flags__context__content"
                    })
                }
                render() {
                    const {
                        heapProps: e,
                        className: t
                    } = this.props, {
                        shownBoxType: r
                    } = this.state, {
                        envInfo: {
                            isMobile: n
                        }
                    } = this.context, i = this.renderFlags();
                    if (!i) return null;
                    const o = p.createElement(a.b, {
                        onUpdate: this.setAnalyticsCallbacks
                    }, p.createElement("div", j({
                        className: d()("cl-paper-flags", t),
                        "data-test-id": "citation"
                    }, e ? Object(s.a)(e) : {}), p.createElement("div", {
                        className: "citation__body"
                    }, i), p.createElement(g.a, null, p.createElement("div", {
                        className: d()({
                            "cl-paper-flags__context": !0,
                            "cl-paper-flags__context__is-visible": !!r
                        })
                    }, this.renderBoxContext(r)))));
                    return n ? o : p.createElement(y.a.Consumer, null, e => {
                        let {
                            setExpandedContent: t
                        } = e;
                        return this._setControlsExpandedContent = t, o
                    })
                }
            }
            T(R, "defaultProps", {
                excerpts: !0,
                shouldRenderIntents: !1
            }), T(R, "contextTypes", {
                envInfo: x.a.instanceOf(w.a).isRequired
            });
            const L = Object(S.a)({
                    id: "id",
                    title: {
                        text: "title"
                    },
                    slug: "slug",
                    authors: [],
                    venue: {
                        text: "venue"
                    },
                    year: 2018,
                    citationContexts: [{
                        intents: ["methodology"],
                        context: Object(_.b)({
                            text: "Yo, this paper was super good."
                        })
                    }, {
                        intents: ["background"],
                        context: Object(_.b)({
                            text: "Yo, this paper was super good."
                        })
                    }, {
                        intents: ["result"],
                        context: Object(_.b)({
                            text: "Yo, this paper was super good."
                        })
                    }],
                    isKey: !0,
                    badges: []
                }),
                A = {
                    getComponent: async () => R,
                    fields: [{
                        name: "className",
                        desc: "HTML classes to be added to the component",
                        value: {
                            type: "text",
                            default: ""
                        }
                    }],
                    examples: [{
                        title: "Citation flags",
                        desc: "Flags for a citation with all 3 types of excerpt/intents",
                        props: {
                            shouldRenderIntents: !0,
                            citation: L,
                            citationType: l.a.CITING_PAPERS,
                            citedPaperTitle: Object(_.b)({
                                text: "A Paper with Highlight",
                                fragments: [{
                                    start: 2,
                                    end: 7
                                }]
                            })
                        },
                        render: e => p.createElement("div", {
                            style: {
                                width: "600px",
                                padding: "16px"
                            }
                        }, e)
                    }, {
                        title: "Reference flags",
                        desc: "Flags for a reference with all 3 types of excerpt/intents",
                        props: {
                            shouldRenderIntents: !0,
                            citation: L,
                            citationType: l.a.CITED_PAPERS,
                            citedPaperTitle: Object(_.b)({
                                text: "A Paper with Highlight",
                                fragments: [{
                                    start: 2,
                                    end: 7
                                }]
                            })
                        },
                        render: e => p.createElement("div", {
                            style: {
                                width: "600px",
                                padding: "16px"
                            }
                        }, e)
                    }],
                    events: {}
                }
        },
        495: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return h
            }));
            var a = r(44),
                n = r(9),
                i = r(12),
                s = r(82),
                o = r(15),
                l = r(8),
                c = r.n(l);

            function p() {
                return (p = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function u(e, t) {
                if (null == e) return {};
                var r, a, n = function(e, t) {
                    if (null == e) return {};
                    var r, a, n = {},
                        i = Object.keys(e);
                    for (a = 0; a < i.length; a++) r = i[a], t.indexOf(r) >= 0 || (n[r] = e[r]);
                    return n
                }(e, t);
                if (Object.getOwnPropertySymbols) {
                    var i = Object.getOwnPropertySymbols(e);
                    for (a = 0; a < i.length; a++) r = i[a], t.indexOf(r) >= 0 || Object.prototype.propertyIsEnumerable.call(e, r) && (n[r] = e[r])
                }
                return n
            }

            function d(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class h extends o.PureComponent {
                constructor() {
                    super(...arguments), d(this, "onClick", e => {
                        e.preventDefault();
                        const {
                            navId: t,
                            query: r,
                            onClick: s
                        } = this.props, o = document.querySelector("#" + t);
                        if (!o) return void n.a.warn(`Could not find navId "${t}" in the dom`);
                        const {
                            offsetTop: l
                        } = o;
                        if (l >= 0) {
                            const e = l - 100;
                            a.a.hasNativeSmoothScrollSupport() ? window.scrollTo({
                                top: e,
                                behavior: "smooth"
                            }) : window.scrollTo(0, e), o.focus()
                        }
                        const {
                            history: c
                        } = this.context;
                        c.replace(Object(i.f)({
                            path: "",
                            query: r || {},
                            hash: t
                        })), s && s()
                    })
                }
                render() {
                    const e = this.props,
                        {
                            navId: t,
                            children: r,
                            query: a
                        } = e,
                        n = u(e, ["navId", "children", "query"]);
                    return o.createElement("a", p({}, n, {
                        href: "#" + t,
                        onClick: this.onClick
                    }), r)
                }
            }
            d(h, "contextTypes", {
                history: c.a.instanceOf(s.a).isRequired
            })
        },
        519: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return I
            }));
            var a = r(392),
                n = r(400),
                i = r(382),
                s = r(369),
                o = r(129),
                l = r(139),
                c = r(420),
                p = r(60),
                u = r(517),
                d = r(110),
                h = r(43),
                m = r(9),
                b = r(143),
                f = r(82),
                g = r(393),
                y = r(473),
                O = r(38),
                E = r(381),
                v = r(372),
                P = r(368),
                S = r.n(P),
                w = r(0),
                _ = r(8),
                C = r.n(_),
                x = r(15);

            function j() {
                return (j = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function T(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function k(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class I extends x.Component {
                constructor(e) {
                    for (var t = arguments.length, r = new Array(t > 1 ? t - 1 : 0), a = 1; a < t; a++) r[a - 1] = arguments[a];
                    super(e, ...r), k(this, "fetchSuggestions", void 0), k(this, "doFetchSuggestions", () => {
                        const {
                            queryText: e
                        } = this.state, {
                            suggestionType: t
                        } = this.props;
                        if (!(e.length < 3)) switch (t) {
                            case y.a.PAPER_CITATION:
                                this.fetchPaperCitationCompletions();
                                break;
                            case y.a.PAPER_REFERENCE:
                                this.fetchPaperReferenceCompletions();
                                break;
                            case y.a.AUTHOR_PAPER:
                                this.fetchAuthorPaperCompletions();
                                break;
                            default:
                                return void m.a.warn(`Invalid: SuggestionType ${t} is not associated with a completion call.`)
                        }
                    }), k(this, "fetchPaperCitationCompletions", () => {
                        const {
                            paperDetail: e,
                            queryText: t
                        } = this.state, {
                            api: r
                        } = this.context;
                        if (!e) return;
                        const a = e.paper.id;
                        r.fetchCitationCompletions({
                            paperId: a,
                            prefixQuery: t
                        }).then(e => this.setSuggestions(e, !1))
                    }), k(this, "fetchPaperReferenceCompletions", () => {
                        const {
                            paperDetail: e,
                            queryText: t
                        } = this.state, {
                            api: r
                        } = this.context, a = e.paper.id;
                        r.fetchReferenceCompletions({
                            paperId: a,
                            prefixQuery: t
                        }).then(e => this.setSuggestions(e, !1))
                    }), k(this, "fetchAuthorPaperCompletions", () => {
                        const {
                            authorDetail: e,
                            queryText: t
                        } = this.state, {
                            api: r
                        } = this.context;
                        if (!e) return;
                        const a = e.author.id;
                        r.fetchAuthorPaperCompletions({
                            authorId: a,
                            prefixQuery: t
                        }).then(e => this.setSuggestions(e, !0))
                    }), k(this, "setSuggestions", (e, t) => {
                        const {
                            queryText: r
                        } = this.state, a = w.b.List(e.resultData.completions);
                        t && a.map(e => {
                            e.completionType === u.a.AUTHOR.pluralId && (e.completionType = u.a.COAUTHOR.pluralId)
                        }), this.setState(e => {
                            const t = e.queryText === r ? r : e.suggestQueryText;
                            return {
                                suggestions: a,
                                suggestQueryText: t
                            }
                        })
                    }), k(this, "onChangeQueryText", e => {
                        const t = e.currentTarget.value;
                        t.length >= 3 ? this.setState({
                            queryText: t,
                            showSuggestions: !0,
                            selectedSuggestionIndex: null
                        }, this.fetchSuggestions) : this.setState({
                            queryText: t,
                            suggestQueryText: t,
                            showSuggestions: !1,
                            selectedSuggestionIndex: null
                        })
                    }), k(this, "clickSuggestion", () => {
                        this.setState({
                            showSuggestions: !1
                        }, this.toggleFacetToSelectedSuggestion()), this.props.onSearchbarCondense && this.props.onSearchbarCondense()
                    }), k(this, "toggleFacetToSelectedSuggestion", () => {
                        const {
                            selectedSuggestionIndex: e,
                            suggestions: t
                        } = this.state, {
                            injectQueryStore: r
                        } = this.props, {
                            router: a
                        } = this.context;
                        if (null === e) return;
                        const n = e || 0,
                            i = t.get(n),
                            s = i.completionType,
                            o = i.completion.text;
                        switch (s) {
                            case u.a.AUTHOR.pluralId:
                                r.routeToToggleFilter(u.a.AUTHOR.id, o, a);
                                break;
                            case u.a.COAUTHOR.pluralId:
                                r.routeToToggleFilter(u.a.COAUTHOR.id, o, a);
                                break;
                            case u.a.VENUE.id:
                                r.routeToToggleFilter(u.a.VENUE.id, o, a);
                                break;
                            case u.a.FIELDS_OF_STUDY.pluralId: {
                                const e = Object(d.c)(o);
                                r.routeToToggleFilter(u.a.FIELDS_OF_STUDY.pluralId, e.id, a);
                                break
                            }
                        }
                    }), k(this, "onSubmitQueryString", e => {
                        e.preventDefault();
                        const {
                            router: t
                        } = this.context, {
                            injectQueryStore: r,
                            suggestionType: a
                        } = this.props, n = this.state.queryText.trim();
                        r.routeToQueryString(n, t);
                        const i = this.getEventTarget(a);
                        Object(O.a)(g.a.create(i, {
                            queryString: n
                        }))
                    }), k(this, "onKeyDown", e => {
                        const {
                            selectedSuggestionIndex: t,
                            suggestions: r
                        } = this.state, a = t || 0;
                        switch (e.keyCode) {
                            case E.b:
                                null !== t && (e.preventDefault(), this.clickSuggestion());
                                break;
                            case E.h:
                                null !== t && (e.preventDefault(), this.clickSuggestion(), this.setState({
                                    showSuggestions: !1,
                                    selectedSuggestionIndex: null
                                }));
                                break;
                            case E.c:
                                this.setState({
                                    queryText: "",
                                    suggestQueryText: "",
                                    showSuggestions: !1,
                                    selectedSuggestionIndex: null
                                });
                                break;
                            case E.i:
                                this.state.showSuggestions && !r.isEmpty() && (e.preventDefault(), null === t ? this.selectSuggestion(r.size - 1) : 0 === t ? this.selectNone() : this.selectSuggestion(a - 1));
                                break;
                            case E.a:
                                this.state.showSuggestions && !r.isEmpty() && (e.preventDefault(), null === t ? this.selectSuggestion(0) : t === r.size - 1 ? this.selectNone() : this.selectSuggestion(a + 1))
                        }
                    }), k(this, "onFocus", () => {
                        this.props.onSearchbarFocus && this.props.onSearchbarFocus(), this.setState({
                            showSuggestions: !0
                        }, this.fetchSuggestions)
                    }), k(this, "onBlur", () => {
                        this.setState({
                            showSuggestions: !1
                        })
                    });
                    const n = this.getStateFromQueryStore(),
                        {
                            queryText: i
                        } = n;
                    this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? T(Object(r), !0).forEach((function(t) {
                                k(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : T(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, n, {}, this.getStateFromPaperStore(), {}, this.getStateFromAuthorStore(), {
                        suggestQueryText: i,
                        showSuggestions: !1,
                        suggestions: w.b.List(),
                        selectedSuggestionIndex: null
                    }), this.props.injectQueryStore.registerComponent(this, () => {
                        const {
                            queryText: e
                        } = this.getStateFromQueryStore();
                        this.setState({
                            queryText: e,
                            suggestQueryText: e,
                            suggestions: w.b.List()
                        })
                    }), this.context.paperStore.registerComponent(this, () => {
                        this.setState(this.getStateFromPaperStore())
                    }), this.context.authorStore.registerComponent(this, () => {
                        this.setState(this.getStateFromAuthorStore())
                    })
                }
                componentDidMount() {
                    this.fetchSuggestions = Object(c.a)(this.doFetchSuggestions, 400, {
                        leading: !0,
                        trailing: !0
                    })
                }
                componentWillUnmount() {
                    this.fetchSuggestions && this.fetchSuggestions.cancel()
                }
                getStateFromQueryStore() {
                    return {
                        query: this.props.injectQueryStore.getQuery(),
                        queryText: this.props.injectQueryStore.getQuery().queryString || ""
                    }
                }
                getStateFromPaperStore() {
                    return {
                        paperDetail: this.context.paperStore.getPaperDetail()
                    }
                }
                getStateFromAuthorStore() {
                    return {
                        authorDetail: this.context.authorStore.getAuthorDetails()
                    }
                }
                getEventTarget(e) {
                    switch (e) {
                        case y.a.PAPER_CITATION:
                        case y.a.PAPER_REFERENCE:
                            return p.a.PaperDetail.Citations.SEARCH;
                        case y.a.AUTHOR_PAPER:
                            return p.a.AuthorHomePage.Publications.SEARCH;
                        default:
                            return m.a.warn(`Invalid: SuggestionType ${e} is not associated with an event target.`), ""
                    }
                }
                onSuggestionMouseDown(e) {
                    e.preventDefault()
                }
                selectSuggestion(e, t) {
                    this.setState({
                        selectedSuggestionIndex: e
                    }, t)
                }
                selectNone() {
                    this && this.setState({
                        selectedSuggestionIndex: null
                    })
                }
                shouldDisplaySuggestions() {
                    let e = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : this.state;
                    return e.showSuggestions && !e.suggestions.isEmpty()
                }
                renderSuggestions() {
                    const {
                        selectedSuggestionIndex: e,
                        suggestions: t
                    } = this.state;
                    return x.createElement("ul", {
                        className: "suggestion-dropdown-menu",
                        "data-test-id": "citations-autocomplete-suggestions",
                        onMouseLeave: this.selectNone,
                        role: "listbox"
                    }, t.map((r, a) => {
                        let {
                            completionType: n,
                            completion: s
                        } = r;
                        const o = S()("flex-row-vcenter suggestion truncate-line", {
                                cursor: a === e,
                                "border-bottom": a !== t.size - 1
                            }),
                            l = n ? Object(h.c)(e => e.filterBar.shortLabels[n]) : "",
                            c = Object(v.d)({
                                onClick: () => {
                                    this.selectSuggestion(a, () => {
                                        this.clickSuggestion()
                                    })
                                }
                            });
                        return x.createElement("li", j({
                            key: `${s.text} ${a}`,
                            "data-test-id": `citations-${n}-suggestion`,
                            className: o,
                            role: "option",
                            "aria-selected": e === a,
                            onMouseEnter: () => this.selectSuggestion(a),
                            onMouseDown: this.onSuggestionMouseDown
                        }, c), x.createElement(i.c, {
                            className: "suggestion__text truncate-line",
                            field: s
                        }), n && x.createElement("span", {
                            className: "suggestion-text-type"
                        }, " ", l))
                    }))
                }
                render() {
                    const {
                        formId: e,
                        containerClass: t,
                        placeholder: r,
                        isSearchbarFocused: i,
                        suggestionType: o
                    } = this.props, {
                        queryText: l
                    } = this.state;
                    return x.createElement("form", {
                        id: e || "dropdown-filters__search-within-form",
                        className: "dropdown-filters__search-within-form",
                        role: "search",
                        onSubmit: this.onSubmitQueryString
                    }, x.createElement(n.a, {
                        className: t || "search-within"
                    }, x.createElement(a.default, {
                        type: "search",
                        name: "cite_q",
                        id: "search-within-input",
                        className: S()("dropdown-filters__search-within-input", {
                            expanded: i,
                            "author-search": o === y.a.AUTHOR_PAPER
                        }),
                        autoComplete: "off",
                        onChange: this.onChangeQueryText,
                        onFocus: this.onFocus,
                        onBlur: this.onBlur,
                        onKeyDown: this.onKeyDown,
                        placeholder: r,
                        value: l,
                        "data-test-id": "search-within-input"
                    }), x.createElement("button", {
                        "aria-label": Object(h.c)(e => e.appHeader.searchSubmitAriaLabel),
                        className: "form-submit form-submit__icon-text",
                        "data-test-id": "submit-search-within-input"
                    }, x.createElement("div", {
                        className: "flex-row-vcenter"
                    }, x.createElement(s.a, {
                        width: "14",
                        height: "14",
                        icon: "search-small"
                    })))), this.shouldDisplaySuggestions() ? this.renderSuggestions() : null)
                }
            }
            k(I, "contextTypes", {
                api: C.a.instanceOf(o.a).isRequired,
                authorStore: C.a.instanceOf(l.a).isRequired,
                history: C.a.instanceOf(f.a).isRequired,
                paperStore: C.a.instanceOf(b.b).isRequired,
                router: C.a.object.isRequired
            })
        },
        520: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return w
            }));
            var a = r(371),
                n = r(382),
                i = r(375),
                s = r(43),
                o = r(182),
                l = r(61),
                c = r(370),
                p = r(63),
                u = r(372),
                d = r(15),
                h = r.n(d);

            function m() {
                return (m = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function b(e, t, r) {
                ! function(e, t) {
                    if (t.has(e)) throw new TypeError("Cannot initialize the same private elements twice on an object")
                }(e, t), t.set(e, r)
            }

            function f(e, t) {
                return function(e, t) {
                    if (t.get) return t.get.call(e);
                    return t.value
                }(e, function(e, t, r) {
                    if (!t.has(e)) throw new TypeError("attempted to " + r + " private field on non-instance");
                    return t.get(e)
                }(e, t, "get"))
            }
            const g = e => {
                let {
                    author: t,
                    onClick: r,
                    children: a
                } = e;
                return h.a.createElement(c.a, {
                    to: "AUTHOR_PROFILE",
                    params: {
                        authorId: t.alias.ids.first(),
                        slug: t.alias.slug
                    },
                    className: "author-list__link author-list__author-name",
                    onClick: () => r(t.alias.ids.first())
                }, a)
            };
            g.defaultProps = {
                onClick: () => {}
            };
            const y = e => {
                let {
                    amount: t,
                    collapsed: r,
                    onClick: n,
                    ariaProps: i
                } = e;
                return h.a.createElement(a.default, {
                    ariaProps: i,
                    className: "more-authors-label",
                    onClick: n,
                    testId: r ? "author-list-expand" : "author-list-collapse",
                    label: r ? "+" + Object(l.j)(t, "author", "authors", !1) : "less"
                })
            };
            var O, E, v, P = new WeakMap,
                S = new WeakMap;
            class w extends d.PureComponent {
                constructor() {
                    super(...arguments), b(this, P, {
                        writable: !0,
                        value: h.a.createRef()
                    }), b(this, S, {
                        writable: !0,
                        value: h.a.createRef()
                    }), this.state = {
                        collapsed: !0
                    }
                }
                componentDidUpdate(e, t) {
                    const {
                        collapsed: r
                    } = this.state;
                    t.collapsed !== r && Object(u.c)({
                        collapsed: r,
                        listRef: f(this, S),
                        newContentRef: f(this, P),
                        defaultButtonClass: "more-authors-label",
                        expandedTargetClass: "author-list__link"
                    })
                }
                render() {
                    const {
                        max: e,
                        authors: t,
                        shouldLinkToAHP: r,
                        onAuthorClick: a,
                        heapId: o
                    } = this.props, {
                        collapsed: l
                    } = this.state;
                    if (t.isEmpty()) return null;
                    const c = t.size > e,
                        u = c && l ? t.take(2).push(t.last()) : t,
                        d = t.size - 3,
                        b = {
                            "aria-expanded": !l,
                            "aria-label": Object(s.c)(e => e.moreAuthors.expandAriaLabel, d)
                        },
                        O = c ? h.a.createElement(y, {
                            ariaProps: b,
                            onClick: () => this.setState(e => ({
                                collapsed: !e.collapsed
                            })),
                            collapsed: l,
                            amount: d
                        }) : null,
                        E = u.size - 1;
                    return h.a.createElement("span", {
                        ref: f(this, S),
                        className: "author-list"
                    }, u.map((e, t) => h.a.createElement("span", m({
                        key: `${e.alias.ids.join("-")}-${t}`
                    }, Object(i.a)({
                        id: o,
                        "author-id": e.alias.ids.first()
                    }), {
                        "data-test-id": "author-list",
                        ref: 2 !== t || l ? void 0 : f(this, P)
                    }), 0 === t ? null : h.a.createElement("span", {
                        "aria-hidden": "true"
                    }, ", "), O && l && t === E ? h.a.createElement("span", null, O, " ") : null, r && Object(p.b)(e.alias) ? h.a.createElement(g, {
                        author: e,
                        onClick: a
                    }, h.a.createElement(n.c, {
                        field: e.highlightedField
                    })) : h.a.createElement(n.c, {
                        field: e.highlightedField,
                        className: "author-list__author-name"
                    }))), O && !l ? h.a.createElement("span", null, " ", O) : null)
                }
            }
            O = w, E = "defaultProps", v = {
                max: 6,
                shouldLinkToAHP: !0,
                heapId: o.c,
                onAuthorClick: () => {}
            }, E in O ? Object.defineProperty(O, E, {
                value: v,
                enumerable: !0,
                configurable: !0,
                writable: !0
            }) : O[E] = v
        },
        521: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return b
            }));
            var a, n, i, s = r(382),
                o = r(479),
                l = r(61),
                c = r(1),
                p = r(130),
                u = r(15),
                d = r.n(u),
                h = r(8),
                m = r.n(h);
            class b extends d.a.PureComponent {
                stripYearFromVenue(e) {
                    const t = /\b(18|19|20|21|22)\d{2}\b/g.exec(e);
                    if (t) {
                        const r = e,
                            a = [];
                        t.forEach((function(e) {
                            l.g(e) && a.push(e)
                        }));
                        const n = new RegExp(a.join("|"), "gi");
                        return r.replace(n, "").trim()
                    }
                    return e
                }
                render() {
                    const {
                        paper: {
                            venue: e
                        },
                        stripYear: t,
                        textLength: r,
                        className: a,
                        tooltipClassName: n
                    } = this.props, i = ["truncated-venue-tooltip", a].join(" ").trim(), p = ["tooltip--default-size", "venue-tooltip", n].join(" ").trim();
                    if (!e || "string" != typeof e.text || 0 === e.text.length) return null;
                    const u = t ? this.stripYearFromVenue(e.text) : this.props.paper.venue.text,
                        h = e.merge({
                            text: l.m(u, r || c.a.data.MAX_VENUE_LENGTH)
                        });
                    return !this.context.envInfo.isMobile && h.text !== u ? d.a.createElement(o.a, {
                        className: i,
                        tooltipClassName: p,
                        tooltipContent: u,
                        tooltipPosition: "bottom-left"
                    }, d.a.createElement(s.c, {
                        field: h,
                        testId: "venue-metadata"
                    })) : d.a.createElement(s.c, {
                        field: h,
                        testId: "venue-metadata"
                    })
                }
            }
            a = b, n = "contextTypes", i = {
                envInfo: m.a.instanceOf(p.a).isRequired
            }, n in a ? Object.defineProperty(a, n, {
                value: i,
                enumerable: !0,
                configurable: !0,
                writable: !0
            }) : a[n] = i
        },
        525: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return c
            }));
            var a = r(375),
                n = r(372),
                i = r(15),
                s = r(368),
                o = r.n(s);

            function l() {
                return (l = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function c(e) {
                let {
                    title: t,
                    children: r,
                    onClick: s,
                    className: c,
                    heapProps: p
                } = e;
                const u = i.useMemo(() => s ? Object(n.d)({
                    onClick: s
                }) : {}, [s]);
                return i.createElement("div", l({
                    className: o()("cl-paper-flags__content", c)
                }, u, p ? Object(a.a)(p) : {}), t && i.createElement("div", {
                    className: "cl-paper-flags__content-title"
                }, t), r && i.createElement("div", {
                    className: "cl-paper-flags__content-body"
                }, r))
            }
        },
        534: function(e, t, r) {
            "use strict";
            var a = r(24),
                n = r(389),
                i = r(369),
                s = r(44),
                o = r(378),
                l = r(130),
                c = r(60),
                p = r(43);
            var u = r(82),
                d = r(570),
                h = r(38),
                m = r(372),
                b = r(368),
                f = r.n(b),
                g = r(8),
                y = r.n(g),
                O = r(15);

            function E() {
                return (E = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function v(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            r.d(t, "a", (function() {
                return P
            }));
            class P extends O.PureComponent {
                constructor() {
                    super(...arguments), v(this, "onClickEmail", () => this.onClickShareLink(d.a.EMAIL)), v(this, "onClickFacebook", () => this.onClickShareLink(d.a.FACEBOOK)), v(this, "onClickTwitter", () => this.onClickShareLink(d.a.TWITTER)), v(this, "onClickWikipedia", () => this.onClickShareLink(d.a.WIKIPEDIA)), v(this, "onClickDirectLink", () => this.onClickShareLink(d.a.DIRECT_LINK)), v(this, "_onClickKeyDownTwitterProps", Object(m.d)({
                        onClick: this.onClickTwitter
                    })), v(this, "_onClickKeyDownFacebookProps", Object(m.d)({
                        onClick: this.onClickFacebook
                    })), v(this, "_onClickKeyDownWikipediaProps", Object(m.d)({
                        onClick: this.props.corpusId ? this.onClickWikipedia : this.onClickDirectLink
                    })), v(this, "_onClickKeyDownEmailProps", Object(m.d)({
                        onClick: this.onClickEmail
                    })), this.state = {
                        copiedUrl: !1
                    }
                }
                trackShareEvent(e) {
                    const t = o.a.create(c.a.SHARE, {
                        socialType: e.id
                    });
                    Object(h.a)(t)
                }
                copyToClipboard(e) {
                    s.a.copyToClipboard(e), this.setState({
                        copiedUrl: !0
                    })
                }
                onClickShareLink(e) {
                    const {
                        title: t,
                        message: r,
                        corpusId: n,
                        urlOverride: i
                    } = this.props, {
                        history: s
                    } = this.context;
                    this.trackShareEvent(e);
                    const o = s.location.search,
                        l = Object(a.r)(o);
                    delete l.utm_source;
                    const c = Object(a.a)(l),
                        p = `?utm_source=${e.id}${c?"&"+c:""}`,
                        u = i ? `${i}?utm_source=${e.id}` : o ? window.location.href.replace(s.location.search, p) : window.location.href + p,
                        h = encodeURIComponent(u),
                        m = t ? encodeURIComponent(t) : "",
                        b = r ? `${encodeURIComponent(r)}: ${h}` : u;
                    let f = "";
                    switch (e) {
                        case d.a.EMAIL:
                            f = `mailto:?subject=${m}&body=${b}`;
                            break;
                        case d.a.FACEBOOK:
                            f = function(e) {
                                return "https://www.facebook.com/sharer/sharer.php?u=" + e
                            }(h);
                            break;
                        case d.a.WIKIPEDIA:
                            n && (f = `${function(e){return"https://api.semanticscholar.org/CorpusID:"+e}(n)}?utm_source=${e.id}`);
                            break;
                        case d.a.DIRECT_LINK:
                            f = u;
                            break;
                        case d.a.TWITTER:
                            f = function(e, t) {
                                return `https://twitter.com/intent/tweet?url=${e}&text=${t}`
                            }(h, m)
                    }
                    e === d.a.EMAIL ? window.location.href = f : e === d.a.WIKIPEDIA || e === d.a.DIRECT_LINK ? this.copyToClipboard(f) : window.open(f, "_blank", "left=100,top=100,width=800,height=500,toolbar=1,resizable=0,rel=noreferrer")
                }
                render() {
                    const {
                        envInfo: {
                            isMobile: e
                        }
                    } = this.context, {
                        copiedUrl: t
                    } = this.state, r = f()(this.props.className, {
                        "social-share-options__button-container-mobile": e,
                        "social-share-options__button-container": !e
                    }), {
                        label: a
                    } = this.props, s = t ? Object(p.c)(e => e.socialShareOptions.shareButtonLabel.directShareHoverTextCopied) : Object(p.c)(e => e.socialShareOptions.shareButtonLabel.directShareHoverText);
                    return O.createElement("div", {
                        className: r
                    }, O.createElement("div", {
                        className: "social-share-options__share-buttons"
                    }, a && a, O.createElement(i.a, E({
                        icon: "share-twitter",
                        "data-test-id": "social-share-options__share-social-icon__twitter",
                        className: "social-share-options__share-social-icon",
                        role: "button",
                        tabIndex: 0,
                        "aria-label": Object(p.c)(e => e.socialShareOptions.shareButtonLabel.twitterAriaLabel)
                    }, this._onClickKeyDownTwitterProps)), O.createElement(i.a, E({
                        icon: "share-facebook",
                        className: "social-share-options__share-social-icon",
                        "data-test-id": "social-share-options__share-social-icon__facebook",
                        role: "button",
                        tabIndex: 0,
                        "aria-label": Object(p.c)(e => e.socialShareOptions.shareButtonLabel.facebookAriaLabel)
                    }, this._onClickKeyDownFacebookProps)), O.createElement(n.default, {
                        placement: n.PLACEMENT.BOTTOM,
                        tooltipContent: s
                    }, O.createElement(i.a, E({
                        icon: "fa-link-box",
                        className: "social-share-options__share-social-icon",
                        "data-test-id": "social-share-options__share-social-icon__link",
                        role: "button",
                        tabIndex: 0,
                        "aria-label": Object(p.c)(e => e.socialShareOptions.shareButtonLabel.directLinkAriaLabel)
                    }, this._onClickKeyDownWikipediaProps))), O.createElement(i.a, E({
                        icon: "share-email",
                        className: "social-share-options__share-social-icon",
                        "data-test-id": "social-share-options__share-social-icon__email",
                        role: "button",
                        tabIndex: 0,
                        "aria-label": Object(p.c)(e => e.socialShareOptions.shareButtonLabel.emailAriaLabel)
                    }, this._onClickKeyDownEmailProps))))
                }
            }
            v(P, "contextTypes", {
                envInfo: y.a.instanceOf(l.a).isRequired,
                history: y.a.instanceOf(u.a).isRequired
            })
        },
        594: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return u
            }));
            var a = r(159),
                n = r(372),
                i = r(368),
                s = r.n(i),
                o = r(15),
                l = r.n(o);

            function c() {
                return (c = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function p(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class u extends l.a.PureComponent {
                constructor() {
                    super(...arguments), p(this, "onClick", () => {
                        this.state.showMenu || this.setState({
                            showMenu: !this.state.showMenu
                        })
                    }), p(this, "_handleBlurDropdownMenu", e => {
                        if (this.state.showMenu) {
                            let t = e.target,
                                r = !1;
                            for (; t && t instanceof Element && t !== document.body && !r;) r = t === this.refs.dropdown, r || (t = t.parentElement);
                            r || this.setState({
                                showMenu: !1
                            })
                        }
                    }), p(this, "_onClickKeyDownProps", Object(n.d)({
                        onClick: this.onClick
                    })), this.state = {
                        showMenu: !!this.props.showMenu
                    }
                }
                componentWillReceiveProps(e) {
                    (!1 === e.showMenu || !0 === e.showMenu) && this.setState({
                        showMenu: !1
                    })
                }
                componentDidMount() {
                    document.body.addEventListener("click", this._handleBlurDropdownMenu), document.body.addEventListener("keyup", this._handleBlurDropdownMenu)
                }
                componentWillUnmount() {
                    document.body.removeEventListener("click", this._handleBlurDropdownMenu), document.body.removeEventListener("keyup", this._handleBlurDropdownMenu)
                }
                componentDidUpdate(e, t) {
                    t.showMenu !== this.state.showMenu && (this.state.showMenu && this.props.onShow ? this.props.onShow() : this.props.onHide && this.props.onHide())
                }
                render() {
                    const {
                        children: e,
                        content: t,
                        className: r,
                        testId: n
                    } = this.props, {
                        showMenu: i
                    } = this.state, o = e ? l.a.cloneElement(e, {
                        onClick: Object(a.a)(e.props.onClick, () => this.setState({
                            showMenu: !i
                        })),
                        className: s()(e.props.className, "dropdown-menu-trigger")
                    }) : null, p = i ? l.a.createElement("div", {
                        className: "dropdown-menu-content"
                    }, t) : null, u = s()("dropdown-menu", {
                        "dropdown-is-visible": i
                    }, r);
                    return l.a.createElement("div", c({
                        className: u,
                        ref: "dropdown",
                        tabIndex: 0,
                        role: "button",
                        "aria-label": this.props.ariaLabel,
                        "aria-expanded": this.state.showMenu
                    }, this._onClickKeyDownProps, n ? {
                        "data-test-id": n
                    } : {}), o, p)
                }
            }
        },
        598: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return d
            }));
            var a = r(466),
                n = r(369),
                i = r(370),
                s = r(372),
                o = r(368),
                l = r.n(o),
                c = r(15);

            function p() {
                return (p = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function u(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class d extends c.PureComponent {
                constructor() {
                    super(...arguments), u(this, "onToggleClick", () => {
                        this.setState(e => {
                            let {
                                isCollapsed: t
                            } = e;
                            return {
                                isCollapsed: !t
                            }
                        }), this.props.onToggle && this.props.onToggle()
                    }), u(this, "_onClickKeyDownProps", Object(s.d)({
                        onClick: this.onToggleClick
                    })), this.state = {
                        isCollapsed: !(!this.props.isCollapsible || this.props.startOpen)
                    }
                }
                renderTitle() {
                    const {
                        icon: e,
                        title: t
                    } = this.props;
                    return c.createElement("span", {
                        className: "info-box__title flex-row-vcenter"
                    }, e ? c.createElement(n.a, {
                        icon: e,
                        className: "flex-static",
                        width: "25",
                        height: "25"
                    }) : null, c.createElement("h3", null, t))
                }
                renderHeader() {
                    const {
                        isCollapsible: e,
                        title: t,
                        titleLinkProps: r
                    } = this.props, {
                        isCollapsed: n
                    } = this.state, s = n ? a.a.Direction.DOWN : a.a.Direction.UP;
                    return t ? c.createElement("div", p({
                        className: "info-box__header flex-row-vcenter flex-space-between"
                    }, this._onClickKeyDownProps, {
                        role: "button",
                        tabIndex: "0"
                    }), r ? c.createElement(i.a, r, this.renderTitle()) : this.renderTitle(), e ? c.createElement("span", {
                        className: "info-box__toggle"
                    }, c.createElement(a.a, {
                        className: "flex-static",
                        direction: s,
                        width: "14",
                        height: "14"
                    })) : null) : null
                }
                render() {
                    const {
                        className: e,
                        children: t
                    } = this.props, {
                        isCollapsed: r
                    } = this.state, a = l()("info-box", e);
                    return c.createElement("div", {
                        className: a
                    }, this.renderHeader(), r ? null : c.createElement("div", {
                        className: "info-box__content"
                    }, t))
                }
            }
            u(d, "defaultProps", {
                isCollapsible: !1,
                startOpen: !1
            })
        },
        602: function(e, t, r) {
            "use strict";
            r.d(t, "a", (function() {
                return a
            })), r.d(t, "b", (function() {
                return n
            }));
            const a = "home",
                n = "paper_detail"
        },
        604: function(e, t, r) {
            "use strict";
            e.exports = r(605)
        },
        605: function(e, t, r) {
            "use strict";
            /** @license React v16.9.0
             * react-dom-server.browser.production.min.js
             *
             * Copyright (c) Facebook, Inc. and its affiliates.
             *
             * This source code is licensed under the MIT license found in the
             * LICENSE file in the root directory of this source tree.
             */
            var a = r(191),
                n = r(15);

            function i(e) {
                for (var t = e.message, r = "https://reactjs.org/docs/error-decoder.html?invariant=" + t, a = 1; a < arguments.length; a++) r += "&args[]=" + encodeURIComponent(arguments[a]);
                return e.message = "Minified React error #" + t + "; visit " + r + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings. ", e
            }
            var s = "function" == typeof Symbol && Symbol.for,
                o = s ? Symbol.for("react.portal") : 60106,
                l = s ? Symbol.for("react.fragment") : 60107,
                c = s ? Symbol.for("react.strict_mode") : 60108,
                p = s ? Symbol.for("react.profiler") : 60114,
                u = s ? Symbol.for("react.provider") : 60109,
                d = s ? Symbol.for("react.context") : 60110,
                h = s ? Symbol.for("react.concurrent_mode") : 60111,
                m = s ? Symbol.for("react.forward_ref") : 60112,
                b = s ? Symbol.for("react.suspense") : 60113,
                f = s ? Symbol.for("react.suspense_list") : 60120,
                g = s ? Symbol.for("react.memo") : 60115,
                y = s ? Symbol.for("react.lazy") : 60116,
                O = s ? Symbol.for("react.fundamental") : 60117;

            function E(e) {
                if (null == e) return null;
                if ("function" == typeof e) return e.displayName || e.name || null;
                if ("string" == typeof e) return e;
                switch (e) {
                    case l:
                        return "Fragment";
                    case o:
                        return "Portal";
                    case p:
                        return "Profiler";
                    case c:
                        return "StrictMode";
                    case b:
                        return "Suspense";
                    case f:
                        return "SuspenseList"
                }
                if ("object" == typeof e) switch (e.$$typeof) {
                    case d:
                        return "Context.Consumer";
                    case u:
                        return "Context.Provider";
                    case m:
                        var t = e.render;
                        return t = t.displayName || t.name || "", e.displayName || ("" !== t ? "ForwardRef(" + t + ")" : "ForwardRef");
                    case g:
                        return E(e.type);
                    case y:
                        if (e = 1 === e._status ? e._result : null) return E(e)
                }
                return null
            }
            var v = n.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;
            v.hasOwnProperty("ReactCurrentDispatcher") || (v.ReactCurrentDispatcher = {
                current: null
            }), v.hasOwnProperty("ReactCurrentBatchConfig") || (v.ReactCurrentBatchConfig = {
                suspense: null
            });
            var P = {};

            function S(e, t) {
                for (var r = 0 | e._threadCount; r <= t; r++) e[r] = e._currentValue2, e._threadCount = r + 1
            }
            for (var w = new Uint16Array(16), _ = 0; 15 > _; _++) w[_] = _ + 1;
            w[15] = 0;
            var C = /^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,
                x = Object.prototype.hasOwnProperty,
                j = {},
                T = {};

            function k(e) {
                return !!x.call(T, e) || !x.call(j, e) && (C.test(e) ? T[e] = !0 : (j[e] = !0, !1))
            }

            function I(e, t, r, a, n, i) {
                this.acceptsBooleans = 2 === t || 3 === t || 4 === t, this.attributeName = a, this.attributeNamespace = n, this.mustUseProperty = r, this.propertyName = e, this.type = t, this.sanitizeURL = i
            }
            var N = {};
            "children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach((function(e) {
                N[e] = new I(e, 0, !1, e, null, !1)
            })), [
                ["acceptCharset", "accept-charset"],
                ["className", "class"],
                ["htmlFor", "for"],
                ["httpEquiv", "http-equiv"]
            ].forEach((function(e) {
                var t = e[0];
                N[t] = new I(t, 1, !1, e[1], null, !1)
            })), ["contentEditable", "draggable", "spellCheck", "value"].forEach((function(e) {
                N[e] = new I(e, 2, !1, e.toLowerCase(), null, !1)
            })), ["autoReverse", "externalResourcesRequired", "focusable", "preserveAlpha"].forEach((function(e) {
                N[e] = new I(e, 2, !1, e, null, !1)
            })), "allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach((function(e) {
                N[e] = new I(e, 3, !1, e.toLowerCase(), null, !1)
            })), ["checked", "multiple", "muted", "selected"].forEach((function(e) {
                N[e] = new I(e, 3, !0, e, null, !1)
            })), ["capture", "download"].forEach((function(e) {
                N[e] = new I(e, 4, !1, e, null, !1)
            })), ["cols", "rows", "size", "span"].forEach((function(e) {
                N[e] = new I(e, 6, !1, e, null, !1)
            })), ["rowSpan", "start"].forEach((function(e) {
                N[e] = new I(e, 5, !1, e.toLowerCase(), null, !1)
            }));
            var D = /[\-:]([a-z])/g;

            function R(e) {
                return e[1].toUpperCase()
            }
            "accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach((function(e) {
                var t = e.replace(D, R);
                N[t] = new I(t, 1, !1, e, null, !1)
            })), "xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach((function(e) {
                var t = e.replace(D, R);
                N[t] = new I(t, 1, !1, e, "http://www.w3.org/1999/xlink", !1)
            })), ["xml:base", "xml:lang", "xml:space"].forEach((function(e) {
                var t = e.replace(D, R);
                N[t] = new I(t, 1, !1, e, "http://www.w3.org/XML/1998/namespace", !1)
            })), ["tabIndex", "crossOrigin"].forEach((function(e) {
                N[e] = new I(e, 1, !1, e.toLowerCase(), null, !1)
            })), N.xlinkHref = new I("xlinkHref", 1, !1, "xlink:href", "http://www.w3.org/1999/xlink", !0), ["src", "href", "action", "formAction"].forEach((function(e) {
                N[e] = new I(e, 1, !1, e.toLowerCase(), null, !0)
            }));
            var L = /["'&<>]/;

            function A(e) {
                if ("boolean" == typeof e || "number" == typeof e) return "" + e;
                e = "" + e;
                var t = L.exec(e);
                if (t) {
                    var r, a = "",
                        n = 0;
                    for (r = t.index; r < e.length; r++) {
                        switch (e.charCodeAt(r)) {
                            case 34:
                                t = "&quot;";
                                break;
                            case 38:
                                t = "&amp;";
                                break;
                            case 39:
                                t = "&#x27;";
                                break;
                            case 60:
                                t = "&lt;";
                                break;
                            case 62:
                                t = "&gt;";
                                break;
                            default:
                                continue
                        }
                        n !== r && (a += e.substring(n, r)), n = r + 1, a += t
                    }
                    e = n !== r ? a + e.substring(n, r) : a
                }
                return e
            }

            function F(e, t) {
                var r, a = N.hasOwnProperty(e) ? N[e] : null;
                return (r = "style" !== e) && (r = null !== a ? 0 === a.type : 2 < e.length && ("o" === e[0] || "O" === e[0]) && ("n" === e[1] || "N" === e[1])), r || function(e, t, r, a) {
                    if (null == t || function(e, t, r, a) {
                            if (null !== r && 0 === r.type) return !1;
                            switch (typeof t) {
                                case "function":
                                case "symbol":
                                    return !0;
                                case "boolean":
                                    return !a && (null !== r ? !r.acceptsBooleans : "data-" !== (e = e.toLowerCase().slice(0, 5)) && "aria-" !== e);
                                default:
                                    return !1
                            }
                        }(e, t, r, a)) return !0;
                    if (a) return !1;
                    if (null !== r) switch (r.type) {
                        case 3:
                            return !t;
                        case 4:
                            return !1 === t;
                        case 5:
                            return isNaN(t);
                        case 6:
                            return isNaN(t) || 1 > t
                    }
                    return !1
                }(e, t, a, !1) ? "" : null !== a ? (e = a.attributeName, 3 === (r = a.type) || 4 === r && !0 === t ? e + '=""' : (a.sanitizeURL && (t = "" + t), e + '="' + A(t) + '"')) : k(e) ? e + '="' + A(t) + '"' : ""
            }
            var M = null,
                q = null,
                B = null,
                V = !1,
                H = !1,
                Q = null,
                U = 0;

            function z() {
                if (null === M) throw i(Error(321));
                return M
            }

            function Y() {
                if (0 < U) throw i(Error(312));
                return {
                    memoizedState: null,
                    queue: null,
                    next: null
                }
            }

            function W() {
                return null === B ? null === q ? (V = !1, q = B = Y()) : (V = !0, B = q) : null === B.next ? (V = !1, B = B.next = Y()) : (V = !0, B = B.next), B
            }

            function G(e, t, r, a) {
                for (; H;) H = !1, U += 1, B = null, r = e(t, a);
                return q = M = null, U = 0, B = Q = null, r
            }

            function K(e, t) {
                return "function" == typeof t ? t(e) : t
            }

            function $(e, t, r) {
                if (M = z(), B = W(), V) {
                    var a = B.queue;
                    if (t = a.dispatch, null !== Q && void 0 !== (r = Q.get(a))) {
                        Q.delete(a), a = B.memoizedState;
                        do {
                            a = e(a, r.action), r = r.next
                        } while (null !== r);
                        return B.memoizedState = a, [a, t]
                    }
                    return [B.memoizedState, t]
                }
                return e = e === K ? "function" == typeof t ? t() : t : void 0 !== r ? r(t) : t, B.memoizedState = e, e = (e = B.queue = {
                    last: null,
                    dispatch: null
                }).dispatch = X.bind(null, M, e), [B.memoizedState, e]
            }

            function X(e, t, r) {
                if (!(25 > U)) throw i(Error(301));
                if (e === M)
                    if (H = !0, e = {
                            action: r,
                            next: null
                        }, null === Q && (Q = new Map), void 0 === (r = Q.get(t))) Q.set(t, e);
                    else {
                        for (t = r; null !== t.next;) t = t.next;
                        t.next = e
                    }
            }

            function Z() {}
            var J = 0,
                ee = {
                    readContext: function(e) {
                        var t = J;
                        return S(e, t), e[t]
                    },
                    useContext: function(e) {
                        z();
                        var t = J;
                        return S(e, t), e[t]
                    },
                    useMemo: function(e, t) {
                        if (M = z(), t = void 0 === t ? null : t, null !== (B = W())) {
                            var r = B.memoizedState;
                            if (null !== r && null !== t) {
                                e: {
                                    var a = r[1];
                                    if (null === a) a = !1;
                                    else {
                                        for (var n = 0; n < a.length && n < t.length; n++) {
                                            var i = t[n],
                                                s = a[n];
                                            if ((i !== s || 0 === i && 1 / i != 1 / s) && (i == i || s == s)) {
                                                a = !1;
                                                break e
                                            }
                                        }
                                        a = !0
                                    }
                                }
                                if (a) return r[0]
                            }
                        }
                        return e = e(), B.memoizedState = [e, t], e
                    },
                    useReducer: $,
                    useRef: function(e) {
                        M = z();
                        var t = (B = W()).memoizedState;
                        return null === t ? (e = {
                            current: e
                        }, B.memoizedState = e) : t
                    },
                    useState: function(e) {
                        return $(K, e)
                    },
                    useLayoutEffect: function() {},
                    useCallback: function(e) {
                        return e
                    },
                    useImperativeHandle: Z,
                    useEffect: Z,
                    useDebugValue: Z,
                    useResponder: function(e, t) {
                        return {
                            props: t,
                            responder: e
                        }
                    }
                },
                te = "http://www.w3.org/1999/xhtml";

            function re(e) {
                switch (e) {
                    case "svg":
                        return "http://www.w3.org/2000/svg";
                    case "math":
                        return "http://www.w3.org/1998/Math/MathML";
                    default:
                        return "http://www.w3.org/1999/xhtml"
                }
            }
            var ae = {
                    area: !0,
                    base: !0,
                    br: !0,
                    col: !0,
                    embed: !0,
                    hr: !0,
                    img: !0,
                    input: !0,
                    keygen: !0,
                    link: !0,
                    meta: !0,
                    param: !0,
                    source: !0,
                    track: !0,
                    wbr: !0
                },
                ne = a({
                    menuitem: !0
                }, ae),
                ie = {
                    animationIterationCount: !0,
                    borderImageOutset: !0,
                    borderImageSlice: !0,
                    borderImageWidth: !0,
                    boxFlex: !0,
                    boxFlexGroup: !0,
                    boxOrdinalGroup: !0,
                    columnCount: !0,
                    columns: !0,
                    flex: !0,
                    flexGrow: !0,
                    flexPositive: !0,
                    flexShrink: !0,
                    flexNegative: !0,
                    flexOrder: !0,
                    gridArea: !0,
                    gridRow: !0,
                    gridRowEnd: !0,
                    gridRowSpan: !0,
                    gridRowStart: !0,
                    gridColumn: !0,
                    gridColumnEnd: !0,
                    gridColumnSpan: !0,
                    gridColumnStart: !0,
                    fontWeight: !0,
                    lineClamp: !0,
                    lineHeight: !0,
                    opacity: !0,
                    order: !0,
                    orphans: !0,
                    tabSize: !0,
                    widows: !0,
                    zIndex: !0,
                    zoom: !0,
                    fillOpacity: !0,
                    floodOpacity: !0,
                    stopOpacity: !0,
                    strokeDasharray: !0,
                    strokeDashoffset: !0,
                    strokeMiterlimit: !0,
                    strokeOpacity: !0,
                    strokeWidth: !0
                },
                se = ["Webkit", "ms", "Moz", "O"];
            Object.keys(ie).forEach((function(e) {
                se.forEach((function(t) {
                    t = t + e.charAt(0).toUpperCase() + e.substring(1), ie[t] = ie[e]
                }))
            }));
            var oe = /([A-Z])/g,
                le = /^ms-/,
                ce = n.Children.toArray,
                pe = v.ReactCurrentDispatcher,
                ue = {
                    listing: !0,
                    pre: !0,
                    textarea: !0
                },
                de = /^[a-zA-Z][a-zA-Z:_\.\-\d]*$/,
                he = {},
                me = {};
            var be = Object.prototype.hasOwnProperty,
                fe = {
                    children: null,
                    dangerouslySetInnerHTML: null,
                    suppressContentEditableWarning: null,
                    suppressHydrationWarning: null
                };

            function ge(e, t) {
                if (void 0 === e) throw i(Error(152), E(t) || "Component")
            }

            function ye(e, t, r) {
                function s(n, s) {
                    var o = s.prototype && s.prototype.isReactComponent,
                        l = function(e, t, r, a) {
                            if (a && ("object" == typeof(a = e.contextType) && null !== a)) return S(a, r), a[r];
                            if (e = e.contextTypes) {
                                for (var n in r = {}, e) r[n] = t[n];
                                t = r
                            } else t = P;
                            return t
                        }(s, t, r, o),
                        c = [],
                        p = !1,
                        u = {
                            isMounted: function() {
                                return !1
                            },
                            enqueueForceUpdate: function() {
                                if (null === c) return null
                            },
                            enqueueReplaceState: function(e, t) {
                                p = !0, c = [t]
                            },
                            enqueueSetState: function(e, t) {
                                if (null === c) return null;
                                c.push(t)
                            }
                        },
                        d = void 0;
                    if (o) d = new s(n.props, l, u), "function" == typeof s.getDerivedStateFromProps && (null != (o = s.getDerivedStateFromProps.call(null, n.props, d.state)) && (d.state = a({}, d.state, o)));
                    else if (M = {}, d = s(n.props, l, u), null == (d = G(s, n.props, d, l)) || null == d.render) return void ge(e = d, s);
                    if (d.props = n.props, d.context = l, d.updater = u, void 0 === (u = d.state) && (d.state = u = null), "function" == typeof d.UNSAFE_componentWillMount || "function" == typeof d.componentWillMount)
                        if ("function" == typeof d.componentWillMount && "function" != typeof s.getDerivedStateFromProps && d.componentWillMount(), "function" == typeof d.UNSAFE_componentWillMount && "function" != typeof s.getDerivedStateFromProps && d.UNSAFE_componentWillMount(), c.length) {
                            u = c;
                            var h = p;
                            if (c = null, p = !1, h && 1 === u.length) d.state = u[0];
                            else {
                                o = h ? u[0] : d.state;
                                var m = !0;
                                for (h = h ? 1 : 0; h < u.length; h++) {
                                    var b = u[h];
                                    null != (b = "function" == typeof b ? b.call(d, o, n.props, l) : b) && (m ? (m = !1, o = a({}, o, b)) : a(o, b))
                                }
                                d.state = o
                            }
                        } else c = null;
                    if (ge(e = d.render(), s), n = void 0, "function" == typeof d.getChildContext && "object" == typeof(l = s.childContextTypes))
                        for (var f in n = d.getChildContext())
                            if (!(f in l)) throw i(Error(108), E(s) || "Unknown", f);
                    n && (t = a({}, t, n))
                }
                for (; n.isValidElement(e);) {
                    var o = e,
                        l = o.type;
                    if ("function" != typeof l) break;
                    s(o, l)
                }
                return {
                    child: e,
                    context: t
                }
            }
            var Oe = function() {
                    function e(t, r) {
                        if (!(this instanceof e)) throw new TypeError("Cannot call a class as a function");
                        n.isValidElement(t) ? t.type !== l ? t = [t] : (t = t.props.children, t = n.isValidElement(t) ? [t] : ce(t)) : t = ce(t), t = {
                            type: null,
                            domNamespace: te,
                            children: t,
                            childIndex: 0,
                            context: P,
                            footer: ""
                        };
                        var a = w[0];
                        if (0 === a) {
                            var s = w,
                                o = 2 * (a = s.length);
                            if (!(65536 >= o)) throw i(Error(304));
                            var c = new Uint16Array(o);
                            for (c.set(s), (w = c)[0] = a + 1, s = a; s < o - 1; s++) w[s] = s + 1;
                            w[o - 1] = 0
                        } else w[0] = w[a];
                        this.threadID = a, this.stack = [t], this.exhausted = !1, this.currentSelectValue = null, this.previousWasTextNode = !1, this.makeStaticMarkup = r, this.suspenseDepth = 0, this.contextIndex = -1, this.contextStack = [], this.contextValueStack = []
                    }
                    return e.prototype.destroy = function() {
                        if (!this.exhausted) {
                            this.exhausted = !0, this.clearProviders();
                            var e = this.threadID;
                            w[e] = w[0], w[0] = e
                        }
                    }, e.prototype.pushProvider = function(e) {
                        var t = ++this.contextIndex,
                            r = e.type._context,
                            a = this.threadID;
                        S(r, a);
                        var n = r[a];
                        this.contextStack[t] = r, this.contextValueStack[t] = n, r[a] = e.props.value
                    }, e.prototype.popProvider = function() {
                        var e = this.contextIndex,
                            t = this.contextStack[e],
                            r = this.contextValueStack[e];
                        this.contextStack[e] = null, this.contextValueStack[e] = null, this.contextIndex--, t[this.threadID] = r
                    }, e.prototype.clearProviders = function() {
                        for (var e = this.contextIndex; 0 <= e; e--) this.contextStack[e][this.threadID] = this.contextValueStack[e]
                    }, e.prototype.read = function(e) {
                        if (this.exhausted) return null;
                        var t = J;
                        J = this.threadID;
                        var r = pe.current;
                        pe.current = ee;
                        try {
                            for (var a = [""], n = !1; a[0].length < e;) {
                                if (0 === this.stack.length) {
                                    this.exhausted = !0;
                                    var s = this.threadID;
                                    w[s] = w[0], w[0] = s;
                                    break
                                }
                                var o = this.stack[this.stack.length - 1];
                                if (n || o.childIndex >= o.children.length) {
                                    var l = o.footer;
                                    if ("" !== l && (this.previousWasTextNode = !1), this.stack.pop(), "select" === o.type) this.currentSelectValue = null;
                                    else if (null != o.type && null != o.type.type && o.type.type.$$typeof === u) this.popProvider(o.type);
                                    else if (o.type === b) {
                                        this.suspenseDepth--;
                                        var c = a.pop();
                                        if (n) {
                                            n = !1;
                                            var p = o.fallbackFrame;
                                            if (!p) throw i(Error(303));
                                            this.stack.push(p), a[this.suspenseDepth] += "\x3c!--$!--\x3e";
                                            continue
                                        }
                                        a[this.suspenseDepth] += c
                                    }
                                    a[this.suspenseDepth] += l
                                } else {
                                    var d = o.children[o.childIndex++],
                                        h = "";
                                    try {
                                        h += this.render(d, o.context, o.domNamespace)
                                    } catch (e) {
                                        throw e
                                    }
                                    a.length <= this.suspenseDepth && a.push(""), a[this.suspenseDepth] += h
                                }
                            }
                            return a[0]
                        } finally {
                            pe.current = r, J = t
                        }
                    }, e.prototype.render = function(e, t, r) {
                        if ("string" == typeof e || "number" == typeof e) return "" === (r = "" + e) ? "" : this.makeStaticMarkup ? A(r) : this.previousWasTextNode ? "\x3c!-- --\x3e" + A(r) : (this.previousWasTextNode = !0, A(r));
                        if (e = (t = ye(e, t, this.threadID)).child, t = t.context, null === e || !1 === e) return "";
                        if (!n.isValidElement(e)) {
                            if (null != e && null != e.$$typeof) {
                                if ((r = e.$$typeof) === o) throw i(Error(257));
                                throw i(Error(258), r.toString())
                            }
                            return e = ce(e), this.stack.push({
                                type: null,
                                domNamespace: r,
                                children: e,
                                childIndex: 0,
                                context: t,
                                footer: ""
                            }), ""
                        }
                        var s = e.type;
                        if ("string" == typeof s) return this.renderDOM(e, t, r);
                        switch (s) {
                            case c:
                            case h:
                            case p:
                            case f:
                            case l:
                                return e = ce(e.props.children), this.stack.push({
                                    type: null,
                                    domNamespace: r,
                                    children: e,
                                    childIndex: 0,
                                    context: t,
                                    footer: ""
                                }), "";
                            case b:
                                throw i(Error(294))
                        }
                        if ("object" == typeof s && null !== s) switch (s.$$typeof) {
                            case m:
                                M = {};
                                var E = s.render(e.props, e.ref);
                                return E = G(s.render, e.props, E, e.ref), E = ce(E), this.stack.push({
                                    type: null,
                                    domNamespace: r,
                                    children: E,
                                    childIndex: 0,
                                    context: t,
                                    footer: ""
                                }), "";
                            case g:
                                return e = [n.createElement(s.type, a({
                                    ref: e.ref
                                }, e.props))], this.stack.push({
                                    type: null,
                                    domNamespace: r,
                                    children: e,
                                    childIndex: 0,
                                    context: t,
                                    footer: ""
                                }), "";
                            case u:
                                return r = {
                                    type: e,
                                    domNamespace: r,
                                    children: s = ce(e.props.children),
                                    childIndex: 0,
                                    context: t,
                                    footer: ""
                                }, this.pushProvider(e), this.stack.push(r), "";
                            case d:
                                s = e.type, E = e.props;
                                var v = this.threadID;
                                return S(s, v), s = ce(E.children(s[v])), this.stack.push({
                                    type: e,
                                    domNamespace: r,
                                    children: s,
                                    childIndex: 0,
                                    context: t,
                                    footer: ""
                                }), "";
                            case O:
                                throw i(Error(338));
                            case y:
                                throw i(Error(295))
                        }
                        throw i(Error(130), null == s ? s : typeof s, "")
                    }, e.prototype.renderDOM = function(e, t, r) {
                        var s = e.type.toLowerCase();
                        if (r === te && re(s), !he.hasOwnProperty(s)) {
                            if (!de.test(s)) throw i(Error(65), s);
                            he[s] = !0
                        }
                        var o = e.props;
                        if ("input" === s) o = a({
                            type: void 0
                        }, o, {
                            defaultChecked: void 0,
                            defaultValue: void 0,
                            value: null != o.value ? o.value : o.defaultValue,
                            checked: null != o.checked ? o.checked : o.defaultChecked
                        });
                        else if ("textarea" === s) {
                            var l = o.value;
                            if (null == l) {
                                l = o.defaultValue;
                                var c = o.children;
                                if (null != c) {
                                    if (null != l) throw i(Error(92));
                                    if (Array.isArray(c)) {
                                        if (!(1 >= c.length)) throw i(Error(93));
                                        c = c[0]
                                    }
                                    l = "" + c
                                }
                                null == l && (l = "")
                            }
                            o = a({}, o, {
                                value: void 0,
                                children: "" + l
                            })
                        } else if ("select" === s) this.currentSelectValue = null != o.value ? o.value : o.defaultValue, o = a({}, o, {
                            value: void 0
                        });
                        else if ("option" === s) {
                            c = this.currentSelectValue;
                            var p = function(e) {
                                if (null == e) return e;
                                var t = "";
                                return n.Children.forEach(e, (function(e) {
                                    null != e && (t += e)
                                })), t
                            }(o.children);
                            if (null != c) {
                                var u = null != o.value ? o.value + "" : p;
                                if (l = !1, Array.isArray(c)) {
                                    for (var d = 0; d < c.length; d++)
                                        if ("" + c[d] === u) {
                                            l = !0;
                                            break
                                        }
                                } else l = "" + c === u;
                                o = a({
                                    selected: void 0,
                                    children: void 0
                                }, o, {
                                    selected: l,
                                    children: p
                                })
                            }
                        }
                        if (l = o) {
                            if (ne[s] && (null != l.children || null != l.dangerouslySetInnerHTML)) throw i(Error(137), s, "");
                            if (null != l.dangerouslySetInnerHTML) {
                                if (null != l.children) throw i(Error(60));
                                if ("object" != typeof l.dangerouslySetInnerHTML || !("__html" in l.dangerouslySetInnerHTML)) throw i(Error(61))
                            }
                            if (null != l.style && "object" != typeof l.style) throw i(Error(62), "")
                        }
                        for (E in l = o, c = this.makeStaticMarkup, p = 1 === this.stack.length, u = "<" + e.type, l)
                            if (be.call(l, E)) {
                                var h = l[E];
                                if (null != h) {
                                    if ("style" === E) {
                                        d = void 0;
                                        var m = "",
                                            b = "";
                                        for (d in h)
                                            if (h.hasOwnProperty(d)) {
                                                var f = 0 === d.indexOf("--"),
                                                    g = h[d];
                                                if (null != g) {
                                                    if (f) var y = d;
                                                    else if (y = d, me.hasOwnProperty(y)) y = me[y];
                                                    else {
                                                        var O = y.replace(oe, "-$1").toLowerCase().replace(le, "-ms-");
                                                        y = me[y] = O
                                                    }
                                                    m += b + y + ":", b = d, m += f = null == g || "boolean" == typeof g || "" === g ? "" : f || "number" != typeof g || 0 === g || ie.hasOwnProperty(b) && ie[b] ? ("" + g).trim() : g + "px", b = ";"
                                                }
                                            } h = m || null
                                    }
                                    d = null;
                                    e: if (f = s, g = l, -1 === f.indexOf("-")) f = "string" == typeof g.is;
                                        else switch (f) {
                                            case "annotation-xml":
                                            case "color-profile":
                                            case "font-face":
                                            case "font-face-src":
                                            case "font-face-uri":
                                            case "font-face-format":
                                            case "font-face-name":
                                            case "missing-glyph":
                                                f = !1;
                                                break e;
                                            default:
                                                f = !0
                                        }
                                    f ? fe.hasOwnProperty(E) || (d = k(d = E) && null != h ? d + '="' + A(h) + '"' : "") : d = F(E, h), d && (u += " " + d)
                                }
                            } c || p && (u += ' data-reactroot=""');
                        var E = u;
                        l = "", ae.hasOwnProperty(s) ? E += "/>" : (E += ">", l = "</" + e.type + ">");
                        e: {
                            if (null != (c = o.dangerouslySetInnerHTML)) {
                                if (null != c.__html) {
                                    c = c.__html;
                                    break e
                                }
                            } else if ("string" == typeof(c = o.children) || "number" == typeof c) {
                                c = A(c);
                                break e
                            }
                            c = null
                        }
                        return null != c ? (o = [], ue[s] && "\n" === c.charAt(0) && (E += "\n"), E += c) : o = ce(o.children), e = e.type, r = null == r || "http://www.w3.org/1999/xhtml" === r ? re(e) : "http://www.w3.org/2000/svg" === r && "foreignObject" === e ? "http://www.w3.org/1999/xhtml" : r, this.stack.push({
                            domNamespace: r,
                            type: s,
                            children: o,
                            childIndex: 0,
                            context: t,
                            footer: l
                        }), this.previousWasTextNode = !1, E
                    }, e
                }(),
                Ee = {
                    renderToString: function(e) {
                        e = new Oe(e, !1);
                        try {
                            return e.read(1 / 0)
                        } finally {
                            e.destroy()
                        }
                    },
                    renderToStaticMarkup: function(e) {
                        e = new Oe(e, !0);
                        try {
                            return e.read(1 / 0)
                        } finally {
                            e.destroy()
                        }
                    },
                    renderToNodeStream: function() {
                        throw i(Error(207))
                    },
                    renderToStaticNodeStream: function() {
                        throw i(Error(208))
                    },
                    version: "16.9.0"
                },
                ve = {
                    default: Ee
                },
                Pe = ve && Ee || ve;
            e.exports = Pe.default || Pe
        },
        622: function(e, t, r) {
            "use strict";
            var a = r(520),
                n = r(382),
                i = r(43),
                s = r(8),
                o = r.n(s),
                l = r(15),
                c = r.n(l);
            const p = e => {
                let {
                    url: t,
                    doi: r
                } = e;
                return c.a.createElement("section", {
                    className: "doi"
                }, c.a.createElement("span", {
                    className: "doi__label"
                }, Object(i.c)(e => e.paper.meta.doi)), c.a.createElement("a", {
                    className: "doi__link",
                    href: t
                }, r))
            };
            p.propTypes = {
                url: o.a.string.isRequired,
                doi: o.a.string.isRequired
            };
            var u, d, h, m = p,
                b = r(521),
                f = r(133),
                g = r(61),
                y = r(368),
                O = r.n(y);
            r.d(t, "a", (function() {
                return E
            }));
            class E extends l.PureComponent {
                handleClick(e, t) {
                    const r = this.props.onClick;
                    "function" == typeof r && r(e, t, this.props.paper.id)
                }
                renderAuthorsListItem() {
                    const {
                        authors: e
                    } = this.props.paper;
                    return e.isEmpty() ? null : l.createElement("li", {
                        "data-test-id": "author-list"
                    }, l.createElement(a.a, {
                        authors: e,
                        onAuthorClick: e => {
                            this.handleClick("AUTHOR", e)
                        },
                        max: this.props.maxAuthors,
                        shouldLinkToAHP: !this.props.disableAuthorLinks,
                        heapId: this.props.authorHeapId
                    }))
                }
                renderVenueListItem() {
                    const {
                        paper: e,
                        paper: {
                            venue: t
                        }
                    } = this.props;
                    return t && "string" == typeof t.text && 0 !== t.text.length ? l.createElement("li", {
                        key: "venue",
                        "data-test-id": "venue-metadata"
                    }, l.createElement(b.a, {
                        paper: e,
                        stripYear: !0
                    })) : null
                }
                renderYearListItem() {
                    const {
                        year: e
                    } = this.props.paper;
                    return Object(g.g)(e) ? l.createElement("li", {
                        key: "year",
                        "data-test-id": "paper-year"
                    }, "string" == typeof e.text ? l.createElement(n.c, {
                        field: e
                    }) : e) : null
                }
                renderDoiListItem() {
                    const {
                        doiInfo: e
                    } = this.props.paper;
                    return e ? l.createElement("li", {
                        key: "doi",
                        "data-test-id": "paper-doi"
                    }, l.createElement(m, {
                        url: e.doiUrl,
                        doi: e.doi
                    })) : null
                }
                renderCorpusId() {
                    const {
                        corpusId: e
                    } = this.props.paper;
                    return l.createElement("li", {
                        key: "corpus-id",
                        "data-test-id": "corpus-id"
                    }, Object(i.c)(e => e.paper.meta.corpusId, e))
                }
                render() {
                    const {
                        className: e
                    } = this.props;
                    return l.createElement("ul", {
                        className: O()("paper-meta", e),
                        "data-test-id": "paper-meta-subhead"
                    }, !this.props.isPDPMeta && this.renderAuthorsListItem(), this.props.isFreshPDP ? this.renderDoiListItem() : [this.renderVenueListItem(), this.renderYearListItem()], this.renderCorpusId())
                }
            }
            u = E, d = "contextTypes", h = {
                envInfo: o.a.object.isRequired,
                weblabStore: o.a.instanceOf(f.b).isRequired
            }, d in u ? Object.defineProperty(u, d, {
                value: h,
                enumerable: !0,
                configurable: !0,
                writable: !0
            }) : u[d] = h
        },
        732: function(e, t, r) {
            "use strict";
            r.r(t);
            var a = r(477),
                n = r(399),
                i = r(474),
                s = r(376),
                o = r(377),
                l = r(15),
                c = r.n(l);
            class p extends l.PureComponent {
                render() {
                    return l.createElement("div", {
                        className: "card-container"
                    }, this.props.children)
                }
            }
            var u, d, h, m = r(129),
                b = r(130),
                f = r(1),
                g = r(131),
                y = r(8),
                O = r.n(y);

            function E() {
                return (E = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function v(e, t) {
                if (null == e) return {};
                var r, a, n = function(e, t) {
                    if (null == e) return {};
                    var r, a, n = {},
                        i = Object.keys(e);
                    for (a = 0; a < i.length; a++) r = i[a], t.indexOf(r) >= 0 || (n[r] = e[r]);
                    return n
                }(e, t);
                if (Object.getOwnPropertySymbols) {
                    var i = Object.getOwnPropertySymbols(e);
                    for (a = 0; a < i.length; a++) r = i[a], t.indexOf(r) >= 0 || Object.prototype.propertyIsEnumerable.call(e, r) && (n[r] = e[r])
                }
                return n
            }
            class P extends c.a.PureComponent {
                constructor() {
                    super(...arguments);
                    const {
                        dispatcher: e
                    } = this.context, t = {
                        actionType: f.a.actions.PAPER_NAV_TARGET_ADDED,
                        navId: this.props.id,
                        navLabel: this.props.navLabel
                    };
                    e.isDispatching() ? e.dispatchEventually(t) : e.dispatch(t)
                }
                componentWillUnmount() {
                    this.context.dispatcher.dispatchEventually({
                        actionType: f.a.actions.PAPER_NAV_TARGET_REMOVED,
                        navId: this.props.id
                    })
                }
                render() {
                    const e = this.props,
                        {
                            navLabel: t,
                            children: r
                        } = e,
                        a = v(e, ["navLabel", "children"]);
                    return c.a.createElement("div", E({}, a, {
                        tabIndex: 0
                    }), r)
                }
            }
            u = P, d = "contextTypes", h = {
                dispatcher: O.a.instanceOf(g.a).isRequired
            }, d in u ? Object.defineProperty(u, d, {
                value: h,
                enumerable: !0,
                configurable: !0,
                writable: !0
            }) : u[d] = h;
            var S = r(368),
                w = r.n(S);
            class _ extends l.PureComponent {
                render() {
                    const {
                        cardId: e,
                        testId: t,
                        children: r,
                        navLabel: a,
                        className: n
                    } = this.props, i = this.context.envInfo.isMobile, s = w()({
                        card: !0,
                        card__mobile: i
                    }, n);
                    return l.createElement(P, {
                        id: e,
                        "data-test-id": t,
                        navLabel: a,
                        className: s
                    }, r)
                }
            }

            function C(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(_, "contextTypes", {
                envInfo: O.a.instanceOf(b.a).isRequired
            });
            class x extends l.PureComponent {
                render() {
                    const {
                        children: e,
                        willChildrenOwnLayout: t = !1
                    } = this.props, r = this.context.envInfo.isMobile;
                    return l.createElement("div", {
                        className: w()({
                            "card-content": !0,
                            "card-content__mobile": r,
                            "card-content__children-own-layout": t
                        })
                    }, e)
                }
            }
            C(x, "defaultProps", {
                willChildrenOwnLayout: !1
            }), C(x, "contextTypes", {
                envInfo: O.a.instanceOf(b.a).isRequired
            });
            class j extends l.PureComponent {
                render() {
                    const {
                        title: e,
                        subtitle: t
                    } = this.props;
                    return e ? l.createElement("div", {
                        className: "card-header-title"
                    }, l.createElement("h2", {
                        className: "card-header-title__title"
                    }, e), t && l.createElement("h5", {
                        className: "card-header-title__subtitle"
                    }, t)) : null
                }
            }
            class T extends l.PureComponent {
                render() {
                    return l.createElement("div", {
                        className: "card-header-aside"
                    }, this.props.children)
                }
            }
            class k extends l.PureComponent {
                renderChildren() {
                    const {
                        title: e,
                        subtitle: t,
                        children: r
                    } = this.props;
                    return e ? l.createElement(j, {
                        title: e,
                        subtitle: t
                    }) : !!r && r
                }
                render() {
                    const e = this.context.envInfo.isMobile,
                        t = this.renderChildren(),
                        r = !!l.Children.toArray(t).find(e => e.type === T);
                    return l.createElement("div", {
                        className: w()({
                            "card-header": !0,
                            "card-header__mobile": e,
                            "card-header__has-aside": r
                        })
                    }, t)
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(k, "contextTypes", {
                envInfo: O.a.instanceOf(b.a).isRequired
            });
            var I = r(141),
                N = r(371),
                D = r(407),
                R = r(385),
                L = r(22),
                A = r(189),
                F = r(43);
            class M extends l.PureComponent {
                render() {
                    const {
                        onChangeSelected: e,
                        scorecardStats: t,
                        title: r,
                        value: a
                    } = this.props, n = L.b.getIntents().filter(e => {
                        let {
                            id: r
                        } = e;
                        return function(e, t) {
                            return t.some(t => {
                                var r;
                                const a = null != (r = t) && null != (r = r.citationIntentCount) ? r[e] : r;
                                return t.typeKey === A.a && (a && a > 0 || e === L.b.INTENTS.ALL_INTENTS.id)
                            })
                        }(r, t)
                    });
                    return l.createElement("div", {
                        className: "citation-intent-select"
                    }, r && l.createElement("div", {
                        className: "facet-toggle"
                    }, l.createElement("h4", {
                        className: "search-fos-menu__dropdown__title"
                    }, r)), l.createElement("ul", {
                        role: "radiogroup",
                        className: "search-fos-menu__dropdown__link-list"
                    }, n.map(t => {
                        let {
                            id: r
                        } = t;
                        return l.createElement("li", {
                            className: "search-cite-menu__dropdown__link-list-line",
                            key: r
                        }, l.createElement("input", {
                            type: "radio",
                            name: "citeFilter",
                            id: "citeFilter" + r,
                            className: "legacy__input search-cite-menu__dropdown__link-list-line__radio",
                            value: r,
                            onChange: e,
                            checked: r == a,
                            "aria-checked": r === a
                        }), l.createElement("label", {
                            className: "search-cite-menu__dropdown__link-list-line__label",
                            htmlFor: "citeFilter" + r
                        }, Object(F.c)(e => e.filter[r].label)))
                    })))
                }
            }
            var q = r(381),
                B = r(372);

            function V(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class H extends l.PureComponent {
                constructor(e) {
                    for (var t = arguments.length, r = new Array(t > 1 ? t - 1 : 0), a = 1; a < t; a++) r[a - 1] = arguments[a];
                    super(e, ...r), V(this, "onClickCapture", e => {
                        e.preventDefault(), this.setState({
                            isDropdownVisible: !1
                        })
                    }), V(this, "_onClickKeyDownCaptureProps", Object(B.d)({
                        onClick: this.onClickCapture,
                        overrideKeys: [q.c]
                    })), V(this, "onClickDropdown", () => {
                        const {
                            disabled: e
                        } = this.props;
                        e || this.setState({
                            isDropdownVisible: !0
                        })
                    }), V(this, "_onClickKeyDownDropdownProps", Object(B.d)({
                        onClick: this.onClickDropdown
                    })), V(this, "onChangeSelected", e => {
                        const {
                            onChange: t
                        } = this.props;
                        this.setState({
                            isDropdownVisible: !1
                        }), t(e)
                    }), this.state = {
                        isDropdownVisible: !1
                    }
                }
                render() {
                    const {
                        disabled: e,
                        styleAsFacet: t,
                        title: r,
                        useTitleForAll: a,
                        value: n,
                        scorecardStats: i
                    } = this.props, s = L.b.getIntents().find(e => e.id == n), o = s ? Object(F.c)(e => e.filter[s.id].label) : "", c = r && a && "all" === n ? r : o;
                    return l.createElement(D.default, {
                        disabled: e || !s,
                        className: "search-cite-menu__dropdown",
                        label: c,
                        type: t ? N.TYPE.DEFAULT : N.TYPE.PRIMARY,
                        contents: () => l.createElement(M, {
                            scorecardStats: i,
                            onChangeSelected: this.onChangeSelected,
                            title: r,
                            value: n
                        })
                    })
                }
            }
            V(H, "defaultProps", {
                value: L.b.INTENTS.ALL_INTENTS.id
            });
            var Q = r(404),
                U = r(529),
                z = r(530),
                Y = r(531),
                W = r(519),
                G = r(152),
                K = r(60),
                $ = r(110),
                X = r(82),
                Z = r(473),
                J = r(451);
            class ee extends l.PureComponent {
                getCitationQueryStore() {
                    return this.props.citationType === L.a.CITED_PAPERS ? this.context.referenceQueryStore : this.context.citationQueryStore
                }
                render() {
                    const {
                        allFilters: e,
                        filters: t,
                        query: r,
                        scorecardStats: a,
                        stats: n,
                        onChangeCitationIntent: i,
                        onChangeYearFilter: s,
                        onHideClick: o
                    } = this.props, c = this.getCitationQueryStore(), p = this.context.envInfo.isMobile, u = n.get("authors"), d = n.get("venues"), h = Object($.b)(n.get("fieldsOfStudy"));
                    return l.createElement(Q.a, {
                        modalId: null,
                        className: w()({
                            "dropdown-filter-modal": !0,
                            "dropdown-filter-modal__mobile": p
                        }),
                        onHideClick: o
                    }, l.createElement("div", {
                        className: "dropdown-filter-modal__container"
                    }, l.createElement("div", {
                        className: "dropdown-filter-modal__filters"
                    }, l.createElement("div", {
                        className: "dropdown-filter-modal__header"
                    }, e || p ? Object(F.c)(e => e.filterBar.dropdownLabels.allFilters) : Object(F.c)(e => e.filterBar.dropdownLabels.moreFilters)), t.contains(J.b.QUERY_TEXT) && l.createElement(W.a, {
                        formId: "citation-search-text-form-modal",
                        suggestionType: Z.a.PAPER_CITATION,
                        containerClass: "search-within",
                        placeholder: Object(F.c)(e => e.filterCitationBar.searchPlaceholder),
                        injectQueryStore: c
                    }), t.contains(J.b.DATE) && l.createElement(U.a, {
                        analyticsEvent: K.a.Serp.YEAR_SLIDER,
                        centerBucketPopover: !0,
                        filterCallback: s,
                        filters: n,
                        histogramWidth: 200,
                        yearFilter: r.yearFilter,
                        yearBuckets: null,
                        showPresetButtons: !0
                    }), t.contains(J.b.HAS_PDF) && l.createElement(Y.a, {
                        injectQueryStore: c,
                        forCitations: !0
                    }), t.contains(J.b.CITATION_INTENT) && l.createElement(M, {
                        title: "Citation Type",
                        value: r.citationIntent || L.b.INTENTS.ALL_INTENTS.id,
                        onChangeSelected: i,
                        scorecardStats: a
                    }), t.contains(J.b.AUTHOR) && !!u && !u.isEmpty() && l.createElement(z.a, {
                        title: Object(F.c)(e => e.filterBar.dropdownTitles.author),
                        filterType: "author",
                        filters: u,
                        injectQueryStore: c,
                        collapsible: !1
                    }), t.contains(J.b.VENUE) && !!d && !d.isEmpty() && l.createElement(z.a, {
                        title: Object(F.c)(e => e.filterBar.dropdownTitles.venue),
                        filterType: "venue",
                        filters: d,
                        injectQueryStore: c,
                        collapsible: !1,
                        maxLabelLength: f.a.data.MAX_VENUE_LENGTH
                    }), t.contains(J.b.FIELD_OF_STUDY) && !!h && !h.isEmpty() && l.createElement(z.a, {
                        title: Object(F.c)(e => e.filterBar.dropdownTitles.fieldOfStudy),
                        filterType: "fieldsOfStudy",
                        filters: h,
                        collapsible: !1,
                        injectQueryStore: c
                    })), l.createElement("div", {
                        className: "dropdown-filter-modal__footer"
                    }, l.createElement(R.default, {
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.clearFilters),
                        onClick: this.props.onClearFilters
                    }), l.createElement(N.default, {
                        type: "primary",
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.applyFilters),
                        onClick: o
                    }))))
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(ee, "contextTypes", {
                dispatcher: O.a.instanceOf(g.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                router: O.a.object.isRequired,
                citationQueryStore: O.a.instanceOf(I.a),
                referenceQueryStore: O.a.instanceOf(G.a)
            });
            var te = r(400),
                re = r(472),
                ae = r(517),
                ne = r(61),
                ie = r(16),
                se = r(28),
                oe = r(33);

            function le(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function ce(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class pe extends l.Component {
                constructor() {
                    super(...arguments), ce(this, "showAllFilters", () => {
                        this.setState({
                            isAllFiltersVisible: !0
                        })
                    }), ce(this, "showMoreFilters", () => {
                        this.setState({
                            isMoreFiltersVisible: !0
                        })
                    }), ce(this, "showSomeFilters", () => {
                        this.setState({
                            isSomeFiltersVisible: !0
                        })
                    }), ce(this, "hideModals", () => {
                        this.setState({
                            isAllFiltersVisible: !1,
                            isMoreFiltersVisible: !1,
                            isSomeFiltersVisible: !1
                        })
                    }), ce(this, "onSearchbarFocus", () => {
                        this.setState({
                            isSearchbarFocused: !0
                        })
                    }), ce(this, "onSearchbarCondense", () => {
                        const e = document.getElementById("search-within-input");
                        e && e.blur(), this.setState({
                            isSearchbarFocused: !1
                        })
                    }), ce(this, "onChangeYearFilter", (e, t) => {
                        const {
                            router: r
                        } = this.context;
                        this.getCitationQueryStore().routeToYearRange(e, t, r)
                    }), ce(this, "onClearYearFilter", () => {
                        const {
                            router: e
                        } = this.context;
                        this.getCitationQueryStore().routeToYearRangeReset(e)
                    }), ce(this, "onChangeCitationIntent", e => {
                        const t = this.getCitationQueryStore();
                        ie.a.changeRouteForPartialQuery(t.getIndependentQuery(), this.state.query.set("citationIntent", e.currentTarget.value), this.context.history, this.context.router)
                    }), ce(this, "onChangeSort", e => {
                        if ("string" == typeof e) {
                            const {
                                router: t
                            } = this.context;
                            this.getCitationQueryStore().routeToSort(e, t)
                        }
                    }), ce(this, "onClearFilters", () => {
                        const e = this.getCitationQueryStore();
                        this.onSearchbarCondense(), ie.a.changeRouteForPartialQuery(e.getIndependentQuery(), Object(oe.h)(this.state.query), this.context.history, this.context.router)
                    }), ce(this, "isDatesSelected", () => {
                        const {
                            query: e
                        } = this.state, t = e.yearFilter;
                        return !!t && !!t.get("min") && !!t.get("max")
                    }), ce(this, "dateRangeString", () => {
                        const {
                            query: e
                        } = this.state, t = e.yearFilter;
                        if (!t) return "";
                        const r = t.get("min"),
                            a = t.get("max");
                        return r ? r === a ? r.toString() : Object(F.c)(e => e.filterCitationBar.dateRange, r, a || "") : ""
                    }), ce(this, "renderDateClearBox", () => l.createElement("div", {
                        className: "flex-row-vcenter flex-space-between dropdown-filters__clear-date-container"
                    }, l.createElement(R.default, {
                        label: Object(F.c)(e => e.filterCitationBar.clearDate),
                        onClick: this.onClearYearFilter
                    }))), ce(this, "closeDateDropdown", () => {
                        this.setState({
                            isDatePopoverVisible: !1
                        })
                    }), ce(this, "openDateDropdown", () => {
                        this.setState({
                            isDatePopoverVisible: !0
                        })
                    }), ce(this, "getCitationCount", (e, t, r) => r ? e.totalResults : this.props.citationType === L.a.CITED_PAPERS ? t.citedPapers.totalCitations : t.paper.citationStats ? t.paper.citationStats.numCitations : 0), ce(this, "getLabel", (e, t) => e > 9999 && t ? this.props.citationType === L.a.CITED_PAPERS ? Object(F.c)(e => e.filterReferenceBar.moreThanTenThousand) : Object(F.c)(e => e.filterCitationBar.moreThanTenThousand) : e > 1 ? this.props.citationType === L.a.CITED_PAPERS ? Object(F.c)(e => e.filterReferenceBar.exactResultCount, Object(ne.d)(e)) : Object(F.c)(e => e.filterCitationBar.exactResultCount, Object(ne.d)(e)) : 1 === e ? this.props.citationType === L.a.CITED_PAPERS ? Object(F.c)(e => e.filterReferenceBar.singleResultCount) : Object(F.c)(e => e.filterCitationBar.singleResultCount) : this.props.citationType === L.a.CITED_PAPERS ? Object(F.c)(e => e.filterReferenceBar.noResultCount) : Object(F.c)(e => e.filterCitationBar.noResultCount)), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? le(Object(r), !0).forEach((function(t) {
                                ce(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : le(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromCitationQueryStore(), {
                        isDatePopoverVisible: !1,
                        isAllFiltersVisible: !1,
                        isMoreFiltersVisible: !1,
                        isSomeFiltersVisible: !1,
                        isSearchbarFocused: !1
                    });
                    this.getCitationQueryStore().registerComponent(this, () => {
                        this.setState(this.getStateFromCitationQueryStore())
                    })
                }
                getCitationQueryStore() {
                    return this.props.citationType === L.a.CITED_PAPERS ? this.context.referenceQueryStore : this.context.citationQueryStore
                }
                getStateFromCitationQueryStore() {
                    const e = this.getCitationQueryStore();
                    return {
                        query: e.getQuery(),
                        queryText: e.getQuery().queryString || "",
                        queryResponse: e.getQueryResponse(),
                        statsResponse: e.getAggregationResponse(),
                        isLoading: e.isLoading(),
                        isFiltering: e.isFiltering(),
                        isAggsLoading: e.isAggsLoading()
                    }
                }
                render() {
                    const {
                        paperDetail: e,
                        paperDetail: {
                            citationSortAvailability: t,
                            paper: {
                                scorecardStats: r
                            }
                        }
                    } = this.props, {
                        query: a,
                        queryText: n,
                        queryResponse: i,
                        statsResponse: {
                            stats: o
                        },
                        isDatePopoverVisible: c,
                        isLoading: p,
                        isSearchbarFocused: u,
                        isFiltering: d,
                        isAggsLoading: h
                    } = this.state, m = this.getCitationQueryStore(), b = o.get(ae.a.AUTHOR.pluralId), f = a.authors.size > 0, g = a.venues.size > 0, y = a.fieldsOfStudy.size > 0, O = a.requireViewablePdf, E = this.isDatesSelected(), v = a.citationIntent, P = y || g, S = P || v || O || E || y || f, _ = S || n, C = o.get(ae.a.AUTHOR.pluralId) && o.get(ae.a.AUTHOR.pluralId).size > 0, x = o.get(ae.a.VENUE.pluralId) && o.get(ae.a.VENUE.pluralId).size > 0, j = o.get("years") && o.get("years").size > 1, T = C || x, k = this.getCitationCount(i, e, !!_), I = this.getLabel(k, !!_), M = E ? this.dateRangeString() : Object(F.c)(e => e.filterBar.dropdownLabels.date), q = !this.context.envInfo.isMobile, B = j || E, V = d || h || p, Q = (null == t ? void 0 : t.citingPapers) && t.citingPapers.includes(se.a.RELEVANCE.id) ? se.a.citationsFilteredWithRelevance() : se.a.citationsFiltered(), G = function(e) {
                        return e.some(e => {
                            var t, r, a;
                            const n = null == e || null === (t = e.citationIntentCount) || void 0 === t ? void 0 : t.methodology,
                                i = null == e || null === (r = e.citationIntentCount) || void 0 === r ? void 0 : r.background,
                                s = null == e || null === (a = e.citationIntentCount) || void 0 === a ? void 0 : a.result;
                            return e.typeKey === A.a && (n && n > 0 || i && i > 0 || s && s > 0)
                        })
                    }(r), $ = V || 0 === i.totalResults && !v || !G;
                    return l.createElement("div", {
                        className: "dropdown-filters__header dropdown-filters-breakpoints__pdp_cite"
                    }, l.createElement("div", {
                        className: "dropdown-filters__controls"
                    }, p && _ ? l.createElement("div", {
                        className: "dropdown-filters__result-count__header"
                    }, l.createElement("div", {
                        className: "loading-count flex-row-vcenter",
                        role: "main",
                        id: "main-content"
                    }, l.createElement(s.a, {
                        testId: "citation-count-loading"
                    }))) : l.createElement("h2", {
                        className: "dropdown-filters__result-count__header dropdown-filters__result-count__citations"
                    }, I), l.createElement(te.a, {
                        className: w()("flex-row-vcenter", "dropdown-filters__header__outer-flex-container", {
                            "dropdown-filters__header__outer-flex-container--mobile": this.context.envInfo.isMobile
                        })
                    }, l.createElement(te.a, {
                        className: "flex-row-vcenter dropdown-filters__filter-flex-container"
                    }, l.createElement(W.a, {
                        formId: "citation-search-text-form",
                        containerClass: "search-within",
                        placeholder: Object(F.c)(e => e.filterCitationBar.searchPlaceholder),
                        isSearchbarFocused: u,
                        onSearchbarCondense: this.onSearchbarCondense,
                        onSearchbarFocus: this.onSearchbarFocus,
                        suggestionType: this.props.citationType === L.a.CITED_PAPERS ? Z.a.PAPER_REFERENCE : Z.a.PAPER_CITATION,
                        injectQueryStore: m,
                        useOverlay: !0
                    }), u && l.createElement("div", {
                        className: "dropdown-filters__modal-button"
                    }, l.createElement(N.default, {
                        disabled: p,
                        type: N.TYPE.SECONDARY,
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.showFilters),
                        onClick: this.onSearchbarCondense
                    })), !u && l.createElement(l.Fragment, null, B && q && l.createElement(D.default, {
                        disabled: !E && !j || V || $,
                        isDropdownShown: c,
                        onHideDropdown: this.closeDateDropdown,
                        onShowDropdown: this.openDateDropdown,
                        className: "dropdown-filters__dates",
                        label: M,
                        type: E ? N.TYPE.PRIMARY : N.TYPE.DEFAULT,
                        contents: () => l.createElement("div", {
                            className: w()({
                                "dropdown-filters__content--disabled": p
                            })
                        }, l.createElement(U.a, {
                            analyticsEvent: K.a.Serp.YEAR_SLIDER,
                            centerBucketPopover: !0,
                            filterCallback: this.onChangeYearFilter,
                            filters: o,
                            yearFilter: a.yearFilter,
                            yearBuckets: null,
                            showPresetButtons: !0
                        }), E && this.renderDateClearBox())
                    }), q && l.createElement(H, {
                        scorecardStats: r,
                        disabled: $,
                        styleAsFacet: !v,
                        useTitleForAll: !0,
                        title: "Citation Type",
                        dropdownPosition: "bottom-right",
                        value: a.citationIntent,
                        onChange: this.onChangeCitationIntent
                    }), q && l.createElement(Y.a, {
                        isToggle: !0,
                        disabled: V,
                        injectQueryStore: m,
                        forCitations: !0
                    }), q && l.createElement(D.default, {
                        disabled: V || !C,
                        className: "dropdown-filters__author",
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.author),
                        type: f ? N.TYPE.PRIMARY : N.TYPE.DEFAULT,
                        contents: () => l.createElement(z.a, {
                            title: Object(F.c)(e => e.filterBar.dropdownTitles.author),
                            filterType: ae.a.AUTHOR.id,
                            filters: b,
                            collapsible: !1,
                            injectQueryStore: m,
                            hasSelection: f
                        })
                    }), q && l.createElement("div", {
                        className: "dropdown-filters__modal-button flex-item"
                    }, l.createElement(N.default, {
                        disabled: V || !T,
                        className: "dropdown-filters__some_filters",
                        type: P ? N.TYPE.PRIMARY : N.TYPE.SECONDARY,
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.moreFilters),
                        onClick: this.showSomeFilters
                    })), q && l.createElement("div", {
                        className: "dropdown-filters__modal-button flex-item"
                    }, l.createElement(N.default, {
                        disabled: V,
                        className: "dropdown-filters__more_filters",
                        type: S ? N.TYPE.PRIMARY : N.TYPE.SECONDARY,
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.moreFilters),
                        onClick: this.showMoreFilters
                    })), l.createElement("div", {
                        className: "dropdown-filters__modal-button"
                    }, l.createElement(N.default, {
                        disabled: V,
                        className: "dropdown-filters__mobile_filters",
                        type: _ ? N.TYPE.PRIMARY : N.TYPE.SECONDARY,
                        label: Object(F.c)(e => e.filterBar.dropdownLabels.allFilters),
                        onClick: this.showAllFilters
                    }))), _ && l.createElement(R.default, {
                        label: "Clear",
                        onClick: this.onClearFilters
                    })), l.createElement(re.a, {
                        className: "dropdown-filters__sort-control",
                        onChangeSort: this.onChangeSort,
                        options: Q,
                        sort: a.sort
                    }))), this.state.isSomeFiltersVisible && !p && l.createElement(ee, {
                        filters: J.c.startInModal,
                        onChangeCitationIntent: this.onChangeCitationIntent,
                        onChangeYearFilter: this.onChangeYearFilter,
                        onHideClick: this.hideModals,
                        onClearFilters: this.onClearFilters,
                        query: a,
                        scorecardStats: r,
                        stats: o,
                        citationType: this.props.citationType
                    }), this.state.isMoreFiltersVisible && !p && l.createElement(ee, {
                        filters: J.c.startInModal.concat(J.c.collapseToModal),
                        onChangeCitationIntent: this.onChangeCitationIntent,
                        onChangeYearFilter: this.onChangeYearFilter,
                        onHideClick: this.hideModals,
                        onClearFilters: this.onClearFilters,
                        query: a,
                        scorecardStats: r,
                        stats: o,
                        citationType: this.props.citationType
                    }), this.state.isAllFiltersVisible && !p && l.createElement(ee, {
                        filters: J.c.startInModal.concat(J.c.startVisible),
                        onChangeCitationIntent: this.onChangeCitationIntent,
                        onChangeYearFilter: this.onChangeYearFilter,
                        onHideClick: this.hideModals,
                        onClearFilters: this.onClearFilters,
                        query: a,
                        scorecardStats: r,
                        stats: o,
                        citationType: this.props.citationType
                    }))
                }
            }
            ce(pe, "contextTypes", {
                dispatcher: O.a.instanceOf(g.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                citationQueryStore: O.a.instanceOf(I.a),
                referenceQueryStore: O.a.instanceOf(G.a),
                router: O.a.object.isRequired
            });
            var ue = r(369),
                de = r(637),
                he = r(535),
                me = r(525),
                be = r(391),
                fe = r(386),
                ge = r(478),
                ye = r(375),
                Oe = r(182),
                Ee = r(12);

            function ve() {
                return (ve = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function Pe(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Se extends l.PureComponent {
                constructor() {
                    super(...arguments), Pe(this, "showHighlyInfluencedPopover", () => {
                        this.setState({
                            isPopoverVisible: !0
                        })
                    }), Pe(this, "hideHighlyInfluencedPopover", () => {
                        this.setState({
                            isPopoverVisible: !1
                        })
                    }), Pe(this, "renderHighlyInfluencedContext", () => {
                        const {
                            citationType: e,
                            citedPaperTitle: t
                        } = this.props, r = Object(ne.m)(t.text, ge.TRUNCATE_TITLE_LENGTH);
                        return l.createElement(me.a, {
                            title: Object(F.c)(t => t.citations.popup[e].title, r),
                            className: "cl-paper-flags__context__content"
                        }, l.createElement(fe.a, {
                            content: t => t.citations.popup[e].content
                        }))
                    }), Pe(this, "renderHighlyInfluencedCitationBadge", () => {
                        const {
                            citationType: e,
                            isCompact: t
                        } = this.props;
                        return l.createElement(l.Fragment, null, l.createElement(ue.a, {
                            className: "cl-paper-stats__icon",
                            icon: "scorecard-highly-influential",
                            height: "12",
                            width: "12"
                        }), !t && l.createElement("span", {
                            className: "cl-paper-stats__hideable-text"
                        }, Object(F.c)(t => t.citations.highlyInfluenced[e])))
                    }), this.state = {
                        isPopoverVisible: !1
                    }, this.handleBlur = this.handleBlur.bind(this)
                }
                handleBlur(e) {
                    e.currentTarget.contains(e.relatedTarget) || this.setState({
                        isPopoverVisible: !1
                    })
                }
                render() {
                    const {
                        className: e,
                        citation: t,
                        citationType: r,
                        isCompact: a
                    } = this.props, {
                        isPopoverVisible: n
                    } = this.state, {
                        isMobile: i
                    } = this.context.envInfo, s = t.id && Object(Ee.f)({
                        routeName: "PAPER_DETAIL_BY_ID",
                        params: {
                            paperId: t.id
                        },
                        query: {
                            sort: se.a.IS_INFLUENTIAL_CITATION.id
                        },
                        hash: "citing-papers"
                    });
                    return l.createElement("div", ve({
                        className: w()("cl-paper-stat", "cl-paper-stat__collapsable", {
                            "cl-paper-stat__icon-only": a
                        }, "cl-paper-flags__highly-influenced__container", e)
                    }, Object(ye.a)({
                        id: Oe.e,
                        "paper-id": t.id,
                        type: "highlyInfluenced",
                        "citation-type": r
                    }), {
                        "data-test-id": "citingPapers" === r ? "highlyInfluenced" : "highlyInfluential",
                        onFocus: !i && this.showHighlyInfluencedPopover,
                        onBlur: !i && this.handleBlur,
                        onMouseEnter: !i && this.showHighlyInfluencedPopover,
                        onMouseLeave: !i && this.hideHighlyInfluencedPopover
                    }), s ? l.createElement("a", {
                        "aria-expanded": !0,
                        "aria-label": Object(F.c)(e => e.citations.influential.linkAriaLabel),
                        href: s,
                        className: "cl-paper-stats__citation-pdp-link"
                    }, this.renderHighlyInfluencedCitationBadge()) : this.renderHighlyInfluencedCitationBadge(), n && l.createElement(be.default, {
                        className: "cl-paper-flags__popover",
                        arrow: he.a.SIDE_TOP_POS_LEFT
                    }, this.renderHighlyInfluencedContext()))
                }
            }
            Pe(Se, "contextTypes", {
                envInfo: y.instanceOf(b.a).isRequired
            });
            var we = r(409),
                _e = r(64),
                Ce = r(128);

            function xe(e) {
                let {
                    citation: t,
                    citationType: r,
                    citedPaperTitle: a
                } = e;
                const {
                    envInfo: n
                } = Object(Ce.d)(), i = n.isMobile, s = t.id ? Object(Ee.f)({
                    routeName: "PAPER_DETAIL_BY_ID",
                    params: {
                        paperId: t.id
                    },
                    hash: "citing-papers"
                }) : null, o = c.a.createElement(je, {
                    citation: t
                }), l = c.a.createElement(Se, {
                    citation: t,
                    citationType: r,
                    citedPaperTitle: a,
                    className: w()("cl-paper-stats__badge-hover", {
                        "cl-paper-stats__mobile-badge": i
                    })
                }), p = Te(t) && c.a.createElement("li", {
                    "aria-label": Object(ne.j)(t.numCitedBy, Object(F.c)(e => e.citations.citation)),
                    className: "cl-paper-stats__item"
                }, s ? c.a.createElement("a", {
                    "aria-label": Object(F.c)(e => e.citations.citationLinkAriaLabel),
                    href: s,
                    className: "cl-paper-stats__citation-pdp-link"
                }, o) : o), u = t.isKey && c.a.createElement("li", {
                    "aria-label": Object(F.c)(e => e.citations.influential.ariaLabel),
                    className: "cl-paper-stats__item"
                }, l), d = !!t.badges && t.badges.size > 0;
                return p || u || d ? c.a.createElement("ul", {
                    className: "cl-paper-stats-list",
                    "data-test-id": "citation-stats"
                }, p, u, !!d && c.a.createElement("li", {
                    className: "cl-paper-stats__item"
                }, c.a.createElement(de.b, {
                    paper: Object(_e.b)(t),
                    isCompact: !0
                }))) : null
            }

            function je(e) {
                let {
                    citation: t
                } = e;
                const {
                    envInfo: r
                } = Object(Ce.d)(), a = r.isMobile;
                return Te(t) ? c.a.createElement("div", {
                    className: w()("cl-paper-stat", {
                        "cl-paper-stats__mobile-badge": a
                    })
                }, c.a.createElement(ue.a, {
                    icon: "cited-by",
                    width: "12",
                    height: "12",
                    className: "cl-paper-stats__icon"
                }), Object(we.b)(t.numCitedBy), a && Object(ne.j)(t.numCitedBy, Object(F.c)(e => e.citations.citation), null, !0)) : null
            }

            function Te(e) {
                return e.numCitedBy > 0
            }
            var ke = r(390),
                Ie = r(380),
                Ne = r(449),
                De = r(458),
                Re = r(440),
                Le = r(402),
                Ae = r(427),
                Fe = r(413),
                Me = r(394);
            class qe extends l.PureComponent {
                render() {
                    let e = null;
                    return "undefined" != typeof window && window.history && window.history.length > 0 && (e = c.a.createElement("button", {
                        onClick: () => window.history.back(),
                        className: "link-button"
                    }, Object(F.c)(e => e.citations.emptySearchWithin.removeFacetsLabel))), c.a.createElement("div", {
                        className: "empty-pdp-box"
                    }, c.a.createElement("p", null, Object(F.c)(e => e.citations.emptySearchWithin.text)), e)
                }
            }
            var Be = r(396),
                Ve = r(397),
                He = r(44),
                Qe = r(37),
                Ue = r(148),
                ze = r(190);

            function Ye(e) {
                const t = {
                    paperId: e.paperId,
                    page: e.page,
                    section: e.section,
                    sortOrder: e.sortOrder,
                    clickedPosition: e.paperList.findIndex(t => t.id === e.clickedPaper.id),
                    listContext: e.paperList.map(e => ({
                        paperId: e.id,
                        authors: e.authors.map(e => ({
                            name: e.alias.name,
                            ids: e.alias.ids.toJS()
                        })).toJS()
                    })).toJS()
                };
                Object(ze.a)(K.a.PaperDetail.ORDERED_PAPER_LIST_RESULT, t)
            }
            var We = r(10),
                Ge = r(452),
                Ke = r(379),
                $e = r(133);

            function Xe(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Ze(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            const Je = Object(We.taggedSoftError)("filteredCitationsList");
            class et extends c.a.PureComponent {
                constructor() {
                    super(...arguments), Ze(this, "getCitationQueryStore", () => this.props.citationType === L.a.CITED_PAPERS ? this.context.referenceQueryStore : this.context.citationQueryStore), Ze(this, "getStateFromQueryStore", () => {
                        const e = this.getCitationQueryStore();
                        return {
                            citationQueryResponse: e.getQueryResponse(),
                            citationIsLoading: e.isLoading(),
                            citationIsFiltering: e.isFiltering(),
                            citationIsSorting: e.isSorting(),
                            citationQueryState: e.getQueryState(),
                            citationQuery: e.getQuery()
                        }
                    }), Ze(this, "navigateToCitationPage", e => {
                        this.getCitationQueryStore().routeToPage(e, this.context.router), this.props.shouldScrollOnPaginate && this.scrollOnPaginate(this.props.cardId)
                    }), Ze(this, "trackClickCitationLink", e => {
                        const {
                            paper: t,
                            citationType: r
                        } = this.props, {
                            citationQueryResponse: a
                        } = this.state, {
                            page: n,
                            sort: i
                        } = a.query, s = a.results;
                        Ye({
                            paperId: t.id,
                            section: r === L.a.CITING_PAPERS ? "inboundCitations" : "outboundCitations",
                            sortOrder: i,
                            page: n,
                            paperList: s,
                            clickedPaper: e,
                            swapPosition: void 0
                        })
                    }), Ze(this, "scrollOnPaginate", e => {
                        const {
                            paperNavStore: t
                        } = this.context, r = t.getNavTarget(e);
                        r ? He.a.smoothScrollTo(r, () => {}) : Je("Failed to scroll on paginate, getNavTarget did not find target for " + e)
                    }), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? Xe(Object(r), !0).forEach((function(t) {
                                Ze(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Xe(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({
                        swapIndex: null
                    }, this.getStateFromQueryStore(), {}, this.getStateFromWeblabStore());
                    this.getCitationQueryStore().registerComponent(this, () => {
                        this.setState(this.getStateFromQueryStore())
                    }), this.context.weblabStore.registerComponent(this, () => {
                        this.setState(this.getStateFromWeblabStore())
                    })
                }
                getStateFromWeblabStore() {
                    const {
                        weblabStore: e
                    } = this.context;
                    return {
                        isPaperRowV2FontOnly: e.isVariationEnabled(Qe.b.PaperRowV2FontOnly.KEY, Qe.b.PaperRowV2FontOnly.Variation.PAPER_ROW_V2_FONT_ONLY)
                    }
                }
                renderCitationsListWithIntents() {
                    const {
                        paper: e
                    } = this.props, {
                        citationQuery: t,
                        citationQueryResponse: r,
                        isPaperRowV2FontOnly: a
                    } = this.state, n = r.results;
                    if (n.isEmpty()) return null;
                    const i = this.props.citationType;
                    return c.a.createElement("div", {
                        className: "citation-list__citations"
                    }, n.map((r, n) => {
                        const s = Object(_e.b)(r);
                        return c.a.createElement(Ie.c, {
                            key: r.id ? r.id + n.toString() : "",
                            paper: s,
                            eventData: {
                                parentPaper: e,
                                index: n
                            },
                            onClickTitle: !1
                        }, c.a.createElement(Be.a, null, c.a.createElement(Fe.default, {
                            paper: s,
                            className: w()("citation-list__paper-row", {
                                "paper-v2-font-only": a
                            }),
                            title: c.a.createElement(Me.default, {
                                paper: s,
                                onClick: this.trackClickCitationLink,
                                testId: "citation-paper-title",
                                heapProps: {
                                    id: Oe.g,
                                    "paper-id": r.id,
                                    "citation-type": i,
                                    "has-intents": r.citationContexts.size > 0
                                }
                            }),
                            meta: c.a.createElement(Ae.default, {
                                paper: s,
                                authors: !s.authors.isEmpty() && c.a.createElement(Ne.default, {
                                    paper: s,
                                    heapProps: {
                                        id: Oe.d
                                    }
                                })
                            }),
                            controls: c.a.createElement(Le.default, {
                                paper: s,
                                stats: c.a.createElement(xe, {
                                    citation: r,
                                    citationType: i,
                                    citedPaperTitle: e.title
                                }),
                                flags: c.a.createElement(ge.default, {
                                    citation: r,
                                    citationType: i,
                                    citedPaperTitle: e.title,
                                    shouldRenderIntents: !0,
                                    className: "cl-paper-controls__flags"
                                })
                            }),
                            abstract: c.a.createElement(Ge.b, {
                                paper: s,
                                query: t.queryString
                            })
                        })), c.a.createElement(Ve.a, null, c.a.createElement(De.default, {
                            paper: s,
                            title: c.a.createElement(Me.default, {
                                paper: s,
                                onClick: this.trackClickCitationLink,
                                testId: "citation-paper-title",
                                heapProps: {
                                    id: Oe.g,
                                    "paper-id": r.id,
                                    "citation-type": i,
                                    "has-intents": r.citationContexts.size > 0
                                }
                            }),
                            meta: c.a.createElement(Ae.default, {
                                paper: s,
                                shouldStackMeta: !0,
                                authors: !s.authors.isEmpty() && c.a.createElement(Ne.default, {
                                    paper: s,
                                    heapProps: {
                                        id: Oe.d
                                    }
                                })
                            }),
                            controls: r.numCitedBy > 0 && c.a.createElement(Le.default, {
                                paper: s,
                                actions: !1,
                                stats: c.a.createElement(xe, {
                                    citation: r,
                                    citationType: i,
                                    citedPaperTitle: e.title
                                })
                            }),
                            className: w()("citation-list__paper-card", {
                                "paper-v2-font-only": a
                            }),
                            header: c.a.createElement(ge.default, {
                                citation: r,
                                citationType: i,
                                citedPaperTitle: e.title,
                                shouldRenderIntents: !0
                            }),
                            footer: c.a.createElement(Re.a, {
                                paper: s
                            }),
                            abstract: c.a.createElement(Ge.b, {
                                paper: s,
                                query: t.queryString,
                                className: "tldr__paper-card"
                            })
                        })))
                    }))
                }
                renderEmptyMessage() {
                    return c.a.createElement(qe, null)
                }
                render() {
                    const e = this.context.envInfo.isMobile,
                        {
                            citationQueryResponse: t,
                            citationIsFiltering: r,
                            citationIsSorting: a,
                            citationIsLoading: n
                        } = this.state,
                        i = t.totalPages > 1 ? c.a.createElement("div", {
                            className: "citation-pagination flex-row-vcenter"
                        }, n ? c.a.createElement("span", {
                            className: "flex-row-vcenter citation-loading"
                        }, c.a.createElement(s.a, null), " Loading") : null, c.a.createElement(ke.default, {
                            size: e ? ke.SIZE.LARGE : ke.SIZE.DEFAULT,
                            maxVisiblePageButtons: e ? 4 : 5,
                            pageNumber: t.query.page,
                            onPaginate: this.navigateToCitationPage,
                            totalPages: Math.min(L.b.MAX_CITATION_PAGES, t.totalPages)
                        })) : null,
                        o = this.state.citationQueryResponse.totalResults > 0 ? this.renderCitationsListWithIntents() : this.renderEmptyMessage(),
                        l = w()("paper-detail-content-card", "result-page", {
                            "is-filtering": n || r || a
                        });
                    return c.a.createElement("div", {
                        className: l,
                        "data-test-id": this.props.citationType === L.a.CITING_PAPERS ? "cited-by" : "reference"
                    }, o, i)
                }
            }
            Ze(et, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired,
                citationQueryStore: O.a.instanceOf(I.a),
                referenceQueryStore: O.a.instanceOf(G.a),
                envInfo: O.a.instanceOf(b.a).isRequired,
                paperNavStore: O.a.instanceOf(Ue.a).isRequired,
                router: O.a.object.isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var tt = Object(Ke.b)(K.a.PaperDetail.Citations)(et),
                rt = r(420),
                at = r(21),
                nt = r(51),
                it = r(48),
                st = r(31),
                ot = r.n(st);
            class lt extends it.a {
                constructor(e, t) {
                    super(nt.a.SCROLL, ot.a.recursive({
                        target: e
                    }, t))
                }
                static create(e, t) {
                    return new lt(e, t)
                }
            }
            var ct = r(38),
                pt = r(197);

            function ut(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function dt(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class ht extends l.PureComponent {
                constructor() {
                    super(...arguments), dt(this, "setLandmarkRef", e => {
                        this.landmarkRef = e
                    }), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? ut(Object(r), !0).forEach((function(t) {
                                dt(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ut(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({
                        wasSeen: !1,
                        top: 0,
                        bottom: 0
                    }, this.getStateFromWeblabStore()), this.context.weblabStore.registerComponent(this, () => {
                        this.setState(this.getStateFromWeblabStore())
                    }), this.onScroll = Object(rt.a)(this.onScroll.bind(this), 250, {
                        leading: !0,
                        maxWait: 250,
                        trailing: !0
                    }), this.onResize = Object(rt.a)(this.onResize.bind(this), 250, {
                        leading: !0,
                        maxWait: 250,
                        trailing: !0
                    })
                }
                getStateFromWeblabStore() {
                    const {
                        weblabStore: e
                    } = this.context;
                    return {
                        isLoggingEnabled: e.isFeatureEnabled(at.b.LogHeapLandmarks)
                    }
                }
                onScroll() {
                    this.updateOffsets(this.trackLandmark)
                }
                onResize() {
                    this.updateOffsets(this.trackLandmark)
                }
                isInViewport(e) {
                    const {
                        top: t,
                        bottom: r
                    } = this.state;
                    return r >= e.top && t <= e.top + e.height
                }
                trackLandmark() {
                    const {
                        wasSeen: e,
                        isLoggingEnabled: t
                    } = this.state;
                    if (e) return;
                    if (!t) return;
                    const {
                        target: r,
                        section: a
                    } = this.props, n = Object(pt.b)(document);
                    if (this.isInViewport(n) && !e) {
                        const e = +(100 * (n.top / (n.top + n.height + n.bottom) || 0)).toFixed(2);
                        Object(ct.a)(lt.create(r, {
                            section: a,
                            percent: e
                        })), this.setState({
                            wasSeen: !0
                        })
                    }
                }
                updateOffsets(e) {
                    var t;
                    const r = He.a.offsetFromBody(this.landmarkRef) || 0,
                        a = (null != (t = this) && null != (t = t.landmarkRef) ? t.scrollHeight : t) || 0;
                    this.setState({
                        top: r,
                        bottom: r + a
                    }, e)
                }
                componentDidMount() {
                    this.updateOffsets(this.trackLandmark), He.a.listenForScroll(this.onScroll), He.a.listenForResize(this.onResize)
                }
                componentWillUnmount() {
                    He.a.stopListeningForResize(this.onResize), He.a.stopListeningForScroll(this.onScroll)
                }
                render() {
                    const {
                        isLoggingEnabled: e
                    } = this.state;
                    return l.createElement(l.Fragment, null, e && l.createElement("span", {
                        key: "landmark",
                        className: "heap-landmark",
                        ref: this.setLandmarkRef
                    }), l.createElement(l.Fragment, {
                        key: "children"
                    }, this.props.children))
                }
            }

            function mt(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function bt(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            dt(ht, "contextTypes", {
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            class ft extends c.a.PureComponent {
                constructor() {
                    super(...arguments), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? mt(Object(r), !0).forEach((function(t) {
                                bt(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : mt(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromCitationQueryStore(), {}, this.getStateFromPaperNavStore());
                    this.getCitationQueryStore().registerComponent(this, () => {
                        this.setState(this.getStateFromCitationQueryStore())
                    }), this.context.paperNavStore.registerComponent(this, () => {
                        this.setState(this.getStateFromPaperNavStore())
                    })
                }
                getCardId() {
                    return this.props.citationType === L.a.CITED_PAPERS ? "cited-papers" : "citing-papers"
                }
                getCitationQueryStore() {
                    return this.props.citationType === L.a.CITED_PAPERS ? this.context.referenceQueryStore : this.context.citationQueryStore
                }
                getPaperCount() {
                    const e = this.props.paperDetail,
                        t = e.paper.citationStats ? e.paper.citationStats.numCitations : 0;
                    return this.props.citationType === L.a.CITED_PAPERS ? e.citedPapers.totalCitations : t
                }
                getNavLabel() {
                    const e = this.getPaperCount();
                    return this.props.citationType === L.a.CITED_PAPERS ? Object(F.a)(e => e.paperDetail.tabLabels.referencedPapers, e) : Object(F.a)(e => e.paperDetail.tabLabels.citingPapers, e)
                }
                getStateFromCitationQueryStore() {
                    return {
                        isAggregationsLoaded: this.getCitationQueryStore().isAggregationsLoaded()
                    }
                }
                getStateFromPaperNavStore() {
                    const e = this.context.paperNavStore,
                        t = e.navItems.get(this.getCardId());
                    return {
                        isVisibleOrActive: t && (e.isItemVisible(t) || e.isItemActive(t))
                    }
                }
                componentDidUpdate() {
                    const {
                        isVisibleOrActive: e,
                        isAggregationsLoaded: t
                    } = this.state, {
                        paperDetail: {
                            paper: r
                        }
                    } = this.props;
                    r.id && e && !t && setTimeout(() => {
                        this.props.citationType === L.a.CITED_PAPERS ? this.context.api.searchReferencePaperAggregations(r.id) : this.context.api.searchCitationPaperAggregations(r.id)
                    }, 0)
                }
                render() {
                    const {
                        paper: e
                    } = this.props.paperDetail, t = this.getNavLabel(), r = this.getCardId();
                    return c.a.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: Object(F.c)(e => e.paperDetail.sectionTitles.referencedPapers)
                    }, c.a.createElement(_, {
                        cardId: r,
                        navLabel: t,
                        className: r
                    }, c.a.createElement(k, null, c.a.createElement(pe, {
                        paperDetail: this.props.paperDetail,
                        citationType: this.props.citationType
                    })), c.a.createElement(x, null, c.a.createElement(tt, {
                        name: this.props.citationType,
                        cardId: r,
                        shouldScrollOnPaginate: !0,
                        paper: e,
                        citationType: this.props.citationType
                    }))))
                }
            }

            function gt(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function yt(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            bt(ft, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired,
                paperNavStore: O.a.instanceOf(Ue.a).isRequired,
                citationQueryStore: O.a.instanceOf(I.a),
                referenceQueryStore: O.a.instanceOf(G.a)
            });
            class Ot extends l.PureComponent {
                constructor() {
                    super(...arguments), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? gt(Object(r), !0).forEach((function(t) {
                                yt(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : gt(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromCitationQueryStore(), {}, this.getStateFromPaperNavStore()), this.context.citationQueryStore.registerComponent(this, () => {
                        this.setState(this.getStateFromCitationQueryStore())
                    }), this.context.paperNavStore.registerComponent(this, () => {
                        this.setState(this.getStateFromPaperNavStore())
                    })
                }
                getStateFromCitationQueryStore() {
                    return {
                        isAggregationsLoaded: this.context.citationQueryStore.isAggregationsLoaded()
                    }
                }
                getStateFromPaperNavStore() {
                    const e = this.context.paperNavStore,
                        t = e.navItems.get(this.constructor.CARD_ID);
                    return {
                        isVisibleOrActive: t && (e.isItemVisible(t) || e.isItemActive(t))
                    }
                }
                componentDidUpdate() {
                    const {
                        isVisibleOrActive: e,
                        isAggregationsLoaded: t
                    } = this.state, {
                        paperDetail: {
                            paper: r
                        }
                    } = this.props;
                    r.id && e && !t && setTimeout(() => {
                        this.context.api.searchCitationPaperAggregations(r.id)
                    }, 0)
                }
                render() {
                    const {
                        paper: e
                    } = this.props.paperDetail, t = e.citationStats.numCitations, r = Object(F.a)(e => e.paperDetail.tabLabels.citingPapers, t);
                    return l.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: Object(F.c)(e => e.paperDetail.sectionTitles.citingPapers)
                    }, l.createElement(_, {
                        cardId: this.constructor.CARD_ID,
                        navLabel: r,
                        className: "citing-papers"
                    }, l.createElement(k, null, l.createElement(pe, {
                        paperDetail: this.props.paperDetail,
                        citationType: L.a.CITING_PAPERS
                    })), l.createElement(x, null, l.createElement(tt, {
                        name: L.a.CITING_PAPERS,
                        cardId: this.constructor.CARD_ID,
                        shouldScrollOnPaginate: !0,
                        paper: e,
                        citationType: L.a.CITING_PAPERS
                    }))))
                }
            }
            yt(Ot, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired,
                paperNavStore: O.a.instanceOf(Ue.a).isRequired,
                citationQueryStore: O.a.instanceOf(I.a).isRequired
            }), yt(Ot, "CARD_ID", "citing-papers");
            var Et = r(0),
                vt = r(4),
                Pt = r.n(vt);

            function St(e) {
                const {
                    paper: t,
                    paper: {
                        journal: r
                    },
                    skipperExperiments: a
                } = e, n = t.debugInfo;
                if (!n) return null;
                const i = n.sourceIds.map((e, t) => {
                        let r;
                        const a = e.source.id;
                        "DBLP" === a ? r = "http://www.dblp.org/rec/bibtex/" + e.id : "CiteSeerX" === a ? r = "http://citeseerx.ist.psu.edu/viewdoc/summary?doi=" + e.id : "MAG" === a ? r = "https://academic.microsoft.com/paper/" + e.id : "ArXiv" === a && (r = "https://arxiv.org/abs/" + e.id);
                        const n = l.createElement("span", null, l.createElement("span", {
                            className: "source"
                        }, a), ": ", l.createElement("span", {
                            className: "id"
                        }, e.id));
                        let i;
                        return i = void 0 === r ? n : l.createElement("a", {
                            href: r
                        }, n), l.createElement("li", {
                            key: "src-id-" + t,
                            className: "debug-info"
                        }, i)
                    }),
                    s = n.sourceUris.map((e, t) => l.createElement("li", {
                        key: "src-uri-" + t,
                        className: "source-uri debug-info"
                    }, l.createElement("a", {
                        href: e
                    }, e))),
                    o = n.duplicateIds.map((e, t) => l.createElement("li", {
                        key: "dupe-" + t
                    }, "Duplicate ID: ", e)),
                    c = null !== r ? l.createElement("li", {
                        key: "journal-info",
                        className: "debug-info"
                    }, "Journal: ", r.name || "No Journal Name", ", ", r.volume || "Missing Volume", ",", " ", r.pages || "Missing Pages") : l.createElement("li", {
                        key: "journal-info",
                        className: "debug-info"
                    }, "No Journal Info"),
                    p = null != n.earliestAcquisitionDate ? l.createElement("li", {
                        key: "acquisition-date",
                        className: "debug-info"
                    }, "Earliest Acquisition Date:", " ", Pt.a.unix(n.earliestAcquisitionDate).utc().format("MM-DD-YYYY HH:mm z")) : l.createElement("li", {
                        key: "acquisition-date",
                        className: "debug-info"
                    }, "Earliest Acquisition Date: Unknown"),
                    u = l.createElement("li", {
                        key: "fields-of-study",
                        className: "debug-info"
                    }, "Fields Of Study: ", Object(we.e)(t.fieldsOfStudy)),
                    d = l.createElement("li", {
                        key: "corpusId",
                        className: "debug-info"
                    }, "Corpus Id: ", n.corpusId || "Unknown"),
                    h = a ? a.all() : Et.b.List(),
                    m = h.isEmpty() ? null : l.createElement("li", {
                        key: "skipperExperiments",
                        className: "debug-info"
                    }, l.createElement("div", null, "Member of the following Skipper Experiments (", l.createElement("a", {
                        href: "/api/1/skipper/experiments/paper/" + t.id,
                        target: "_blank"
                    }, "see raw data"), "):"), l.createElement("ul", {
                        style: {
                            paddingLeft: "20px"
                        }
                    }, h.map(e => l.createElement("li", {
                        key: e.experimentName
                    }, e.experimentName))));
                return l.createElement("ul", null, i, s, o, c, p, u, d, m)
            }
            class wt extends l.PureComponent {
                render() {
                    const {
                        children: e
                    } = this.props;
                    return l.createElement("div", {
                        className: "card-content-main"
                    }, e)
                }
            }
            const _t = ["250", "500"];

            function Ct(e) {
                let {
                    figure: t
                } = e;
                const r = l.useRef(),
                    [a, n] = l.useState(function(e) {
                        if (!e || e.endsWith("/")) return e;
                        const t = e.lastIndexOf("/"),
                            r = e.slice(0, t) + "/",
                            a = e.slice(t);
                        return _t.map((e, t) => `${r}${e}px${a} ${t+1}x`).join(",")
                    }(t.cdnUri)),
                    i = l.useCallback(() => {
                        n(null)
                    }, []);
                return l.useEffect(() => {
                    const e = r.current;
                    e && e.complete && 0 === e.naturalWidth && n(null)
                }, [r]), l.createElement("img", {
                    className: "figure-list__figure-image",
                    srcSet: a,
                    src: t.cdnUri,
                    onError: i,
                    ref: r,
                    alt: t.displayName,
                    "data-test-id": "figure-thumbnail-img",
                    loading: "lazy"
                })
            }
            var xt = r(378),
                jt = r(7),
                Tt = r(370);
            const kt = "undefined" != typeof ResizeObserver;

            function It(e) {
                let {
                    figures: t,
                    paperId: r,
                    slug: a
                } = e;
                const {
                    envInfo: {
                        isMobile: n
                    }
                } = Object(Ce.d)(), i = n ? 1 : 4, [s, o] = l.useState(!1), [c, p] = l.useState(Object(jt.a)() && !kt || t.size > i), u = l.useCallback(() => {
                    p(!1), o(!0)
                }, []), d = l.useRef();
                return l.useLayoutEffect(() => {
                    if (!kt || !d.current) return;
                    const e = new ResizeObserver(Object(rt.a)(() => {
                        if (s) return void e.disconnect();
                        const t = !![...d.current.children].find(e => 0 === e.clientHeight);
                        t !== c && p(t)
                    }, 100, {
                        maxWait: 100,
                        leading: !0,
                        trailing: !0
                    }));
                    return e.observe(d.current), [...d.current.querySelectorAll(".figure-list__figure_link") || []].forEach(e => {
                        if (c) {
                            const t = e.parentElement.style.overflow;
                            t && "visible" !== t || (e.style.overflow = "hidden");
                            const r = e.parentElement.clientWidth < e.parentElement.scrollWidth || e.parentElement.clientHeight < e.parentElement.scrollHeight;
                            e.parentElement.style.overflow = t, r && (e.tabIndex = -1)
                        } else e.tabIndex = 0
                    }), () => {
                        e.disconnect()
                    }
                }, [d.current, s, c]), t.isEmpty() ? null : l.createElement("div", {
                    className: "figure-list"
                }, l.createElement("ul", {
                    ref: d,
                    className: w()({
                        "figure-list__list": !0,
                        "figure-list__list--is-capped": !s
                    }),
                    "data-test-id": "figure-list"
                }, t.map((e, t) => l.createElement("li", {
                    key: e.uri,
                    className: "figure-list__figure",
                    "data-test-id": "figure-list-item"
                }, l.createElement(Tt.a, {
                    to: "PAPER_DETAIL_FIGURE",
                    className: "figure-list__figure_link",
                    onClick: () => function(e) {
                        let {
                            paperId: t,
                            figureIndex: r
                        } = e;
                        Object(ct.a)(xt.a.create(K.a.FIGURE, {
                            paperId: t,
                            index: r
                        }))
                    }({
                        paperId: r,
                        figureIndex: t
                    }),
                    params: {
                        paperId: r,
                        slug: a,
                        figureIndex: t
                    },
                    tabIndex: 0
                }, l.createElement("figure", null, l.createElement("div", {
                    className: "figure-list__figure-thumb"
                }, l.createElement(Ct, {
                    figure: e
                })), l.createElement("figcaption", {
                    className: "figure-list__figure-caption"
                }, e.displayName)))))), c && l.createElement("div", {
                    className: "figure-list__pagination"
                }, l.createElement(N.default, {
                    onClick: u,
                    label: Object(F.c)(e => e.paperDetail.figureShowMoreLabel, t.size.toLocaleString())
                })))
            }
            class Nt extends l.PureComponent {
                renderFiguresSection(e, t) {
                    return l.createElement(It, {
                        paperId: e.id,
                        slug: e.slug,
                        figures: t
                    })
                }
                renderCardContent(e, t, r) {
                    if (!e.isEmpty()) {
                        const t = l.createElement(l.Fragment, null, !e.isEmpty() && this.renderFiguresSection(r, e));
                        return l.createElement(x, {
                            willChildrenOwnLayout: !0
                        }, l.createElement(wt, null, t))
                    }
                    return null
                }
                hasFigures(e) {
                    return e.some(e => "figure" === e.figureType)
                }
                hasTables(e) {
                    return e.some(e => "table" === e.figureType)
                }
                render() {
                    const {
                        paperDetail: e,
                        paperDetail: {
                            figures: t,
                            paper: r
                        }
                    } = this.props, a = [], n = [];
                    this.hasFigures(t) && (a.push(Object(F.c)(e => e.paperDetail.sectionSubtitles.figures)), n.push(Object(F.c)(e => e.paperDetail.sectionSubtitles.figures))), this.hasTables(t) && (a.push(Object(F.c)(e => e.paperDetail.sectionSubtitles.tables)), n.push(Object(F.c)(e => e.paperDetail.sectionSubtitles.tables)));
                    const i = we.e(a),
                        s = Object(F.c)(e => e.paperDetail.sectionSubtitles.extractedContent, i);
                    if (!t.isEmpty()) {
                        const a = this.renderCardContent(t, e, r);
                        return l.createElement(ht, {
                            target: K.a.PaperDetail.SCROLL_LANDMARKS,
                            section: Object(F.c)(e => e.paperDetail.tabLabels.extractedContent)
                        }, l.createElement(_, {
                            cardId: Nt.CARD_ID,
                            navLabel: i
                        }, l.createElement(k, {
                            title: s
                        }), a && a))
                    }
                    return null
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(Nt, "CARD_ID", "extracted");
            var Dt = r(406),
                Rt = r(501),
                Lt = r(460);

            function At(e) {
                let {
                    dropdownContent: t
                } = e;
                const r = c.a.createElement(be.default, {
                        arrow: be.ARROW_POSITION.SIDE_TOP_POS_RIGHT,
                        className: "reader__dropdown-content"
                    }, t),
                    a = {
                        "aria-label": Object(F.c)(e => e.paperDetail.morePaperLinks)
                    };
                return c.a.createElement(D.default, {
                    className: "reader__dropdown",
                    type: N.TYPE.DEFAULT,
                    children: c.a.createElement(Lt.default, {
                        ariaProps: a,
                        label: null,
                        className: "reader__dropdown-button"
                    }),
                    popover: () => r,
                    usePortal: !1
                })
            }
            var Ft = r(422),
                Mt = r(410);
            class qt extends it.a {
                constructor(e, t) {
                    super(nt.a.CONTEXTMENUOPEN, ot.a.recursive({
                        target: e
                    }, t))
                }
                static create(e, t) {
                    return new qt(e, t)
                }
            }
            var Bt = r(329),
                Vt = r(185);

            function Ht(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Qt(e) {
                for (var t = 1; t < arguments.length; t++) {
                    var r = null != arguments[t] ? arguments[t] : {};
                    t % 2 ? Ht(Object(r), !0).forEach((function(t) {
                        Ut(e, t, r[t])
                    })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ht(Object(r)).forEach((function(t) {
                        Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                    }))
                }
                return e
            }

            function Ut(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }

            function zt(e) {
                const t = qt.create(K.a.PaperDetail.Header.Action.PAPER_LINK, Qt({}, e));
                Object(ct.a)(t)
            }

            function Yt(e) {
                let {
                    link: t,
                    paperId: r,
                    isPrimary: a,
                    onClick: n,
                    entitlement: i,
                    className: s
                } = e;
                const o = Ft.f(t.url),
                    l = t.linkType === Vt.a.OPEN_ACCESS,
                    p = t.linkType === Vt.a.INSTITUTIONAL_ACCESS,
                    u = (() => {
                        if ("entitlement" === t.linkType.toString()) return "getftr";
                        if (i) switch (i.source) {
                            case Bt.a.GETFTR:
                                return "getftr";
                            case Bt.a.LIBKEY:
                            default:
                                return "fa-link-out"
                        } else {
                            if (l) return "open-access";
                            if (o) return "fa-pdf";
                            if (p) return "openathens-login"
                        }
                        return "fa-link-out"
                    })(),
                    d = i ? Object(F.c)(e => e.paper.link[i.source]) : a || p ? Ft.c(t) : "entitlement" === t.linkType ? Object(F.c)(e => e.paper.link.getftr) : Ft.b(t),
                    h = "fa-pdf" === u || "fa-link-out" === u ? "11" : "15",
                    m = {
                        "link-type": t.linkType,
                        "direct-pdf-link": o,
                        "unpaywall-link": l,
                        institutional: p,
                        "primary-link": a,
                        "paper-id": r
                    },
                    b = (() => {
                        if (i) return {
                            entitlement: !0,
                            "entitlement-source": i.source
                        }
                    })(),
                    f = c.a.useCallback(e => {
                        n && n(t, e)
                    }, [t, n]);
                return c.a.createElement(ye.b, {
                    heapProps: Qt({
                        id: Oe.i
                    }, m, {}, b)
                }, c.a.createElement(Mt.a, {
                    tagName: "a",
                    link: t,
                    onClick: f,
                    onContextMenu: () => zt(m),
                    className: w()({
                        "button--full-width": !0,
                        "button--primary": a,
                        "button--unpaywall": l,
                        "button--institutional": p
                    }, s),
                    iconProps: {
                        icon: u,
                        width: h,
                        height: h
                    },
                    testId: "paper-link",
                    text: d,
                    target: "_blank",
                    rel: "noopener",
                    title: t.url,
                    href: t.url
                }))
            }
            var Wt = r(403),
                Gt = r(401),
                Kt = r(111);

            function $t(e, t) {
                var r;
                return !(null == e || null === (r = e.pdfUrl) || void 0 === r || !r.url || t.isMobile)
            }

            function Xt(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Zt(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            var Jt = Object(Ce.c)((function(e) {
                let {
                    createNonBrandedPaperLink: t,
                    paper: r,
                    paperDetail: a,
                    readerVisible: n,
                    envInfo: i
                } = e;
                const s = $t(n, i),
                    o = Ft.h(r, s),
                    l = Ft.g(o),
                    {
                        primaryPaperLink: p
                    } = r,
                    {
                        entitlement: u
                    } = a,
                    d = l.openAccess.isEmpty() ? null : c.a.createElement("ul", {
                        className: "unpaywall-links"
                    }, l.openAccess.map(e => c.a.createElement("li", {
                        key: e.url,
                        className: "unpaywall"
                    }, c.a.createElement(Yt, {
                        link: e,
                        paperId: r.id,
                        isPrimary: !1
                    })))),
                    h = l.pdf.isEmpty() ? null : c.a.createElement("ul", {
                        className: "pdf-links"
                    }, l.pdf.map(e => c.a.createElement("li", {
                        key: e.url,
                        className: "pdf"
                    }, c.a.createElement(Yt, {
                        link: e,
                        paperId: r.id,
                        isPrimary: !1
                    })))),
                    m = l.nonPdf.isEmpty() ? null : c.a.createElement("ul", {
                        className: "out-links"
                    }, l.nonPdf.map(e => c.a.createElement("li", {
                        key: e.url,
                        className: "link-out"
                    }, c.a.createElement(Yt, {
                        link: e,
                        paperId: r.id,
                        isPrimary: !1
                    })))),
                    b = (() => {
                        const e = l.institutional;
                        return e ? c.a.createElement(Wt.a, {
                            feature: at.b.OpenAthensRedirect
                        }, c.a.createElement(Gt.a, null, c.a.createElement("ul", {
                            className: "institutional-link"
                        }, c.a.createElement("li", {
                            key: e.url,
                            className: "institutional"
                        }, c.a.createElement(Yt, {
                            link: e,
                            paperId: r.id,
                            isPrimary: !1
                        }))))) : null
                    })(),
                    f = u ? Object(Kt.b)({
                        url: u.url,
                        linkType: "entitlement",
                        publisherName: "none"
                    }) : p;
                return s && !Ft.e(f) && t && "function" == typeof t ? c.a.createElement(c.a.Fragment, null, t(), d, b, h, m) : c.a.createElement(c.a.Fragment, null, d, b, h, m)
            }), e => {
                const t = Object(Ce.e)(e.paperStore, e => ({
                        paperDetail: e.getPaperDetail()
                    })),
                    r = Object(Ce.e)(e.readerVisibilityStore, e => ({
                        readerVisible: e.getVisibility()
                    }));
                return function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = null != arguments[t] ? arguments[t] : {};
                        t % 2 ? Xt(Object(r), !0).forEach((function(t) {
                            Zt(e, t, r[t])
                        })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Xt(Object(r)).forEach((function(t) {
                            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                        }))
                    }
                    return e
                }({}, {
                    envInfo: e.envInfo
                }, {}, t, {}, r)
            });

            function er(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function tr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            var rr = Object(Ce.c)((function(e) {
                    let {
                        paper: t,
                        readerVisible: r,
                        envInfo: a
                    } = e;
                    const n = Ft.i(t.alternatePaperLinks).first();
                    if (!n) return null;
                    const i = $t(r, a),
                        s = Ft.h(t, i),
                        o = Ft.g(s),
                        l = Ft.d(o),
                        p = n.linkType === Vt.a.OPEN_ACCESS,
                        u = Ft.f(n.url) && !p,
                        d = n.linkType === Vt.a.INSTITUTIONAL_ACCESS,
                        h = Ft.b(n),
                        m = p ? {
                            icon: "open-access",
                            width: "15",
                            height: "15"
                        } : u ? {
                            icon: "pdf",
                            width: "24",
                            height: "18"
                        } : {
                            icon: "fa-link-out",
                            width: "11",
                            height: "11"
                        },
                        b = {
                            "link-type": n.linkType,
                            "direct-pdf-link": Ft.f(n.url),
                            "unpaywall-link": p,
                            institutional: d,
                            "primary-link": !1,
                            "paper-id": t.id
                        };
                    return c.a.createElement(ye.b, {
                        heapProps: {
                            id: Oe.i,
                            "link-type": n.linkType,
                            "direct-pdf-link": Ft.f(n.url),
                            "unpaywall-link": p,
                            institutional: d,
                            "primary-link": !1,
                            "paper-id": t.id
                        }
                    }, c.a.createElement(Mt.a, {
                        tagName: "a",
                        link: n.url,
                        className: w()("flex-paper-actions__button", "alternate-sources__dropdown-button", {
                            "alternate-sources__dropdown-button--show-divider": !l
                        }),
                        iconProps: m,
                        isAlternateSourceButtonLink: !0,
                        isPdf: Ft.f(n.url),
                        onContextMenu: () => zt(b),
                        testId: "paper-link",
                        text: h,
                        target: "_blank",
                        rel: "noopener",
                        title: n.url,
                        href: n.url
                    }))
                }), e => {
                    const t = Object(Ce.e)(e.readerVisibilityStore, e => ({
                        readerVisible: e.getVisibility()
                    }));
                    return function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? er(Object(r), !0).forEach((function(t) {
                                tr(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : er(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, {
                        envInfo: e.envInfo
                    }, {}, t)
                }),
                ar = r(594);

            function nr(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function ir(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            var sr = Object(Ce.c)((function(e) {
                let {
                    createNonBrandedPaperLink: t,
                    envInfo: r,
                    isLoggedIn: a,
                    paper: n,
                    readerVisible: i
                } = e;
                const {
                    weblabStore: s
                } = Object(Ce.d)(), o = $t(i, r), l = Ft.g(n.alternatePaperLinks), p = l.openAccess.isEmpty() && l.pdf.isEmpty() && l.nonPdf.isEmpty(), u = Ft.h(n, o), d = Ft.g(u), h = {
                    id: Oe.a,
                    "paper-id": n.id,
                    "has-open-links": !d.pdf.isEmpty(),
                    "has-open-nonpdf": !d.nonPdf.isEmpty(),
                    "has-unpaywall": !d.openAccess.isEmpty(),
                    "has-institutional": void 0 !== d.institutional,
                    "num-alternates": d.pdf.size + d.nonPdf.size + d.openAccess.size
                };
                if (p) {
                    if (!s.isFeatureEnabled(at.b.OpenAthensRedirect)) return null;
                    const e = l.institutional;
                    if (!e) return null;
                    const t = {
                        id: Oe.i,
                        "link-type": e.linkType,
                        "direct-pdf-link": !1,
                        "unpaywall-link": !1,
                        institutional: !0,
                        "primary-link": !1,
                        "paper-id": n.id
                    };
                    return c.a.createElement(Dt.a, Object(ye.a)(t), c.a.createElement(Yt, {
                        link: e,
                        paperId: n.id,
                        isPrimary: !1,
                        className: "flex-paper-actions__button flex-paper-actions__button--secondary"
                    }))
                }
                const m = a && l.institutional,
                    b = Ft.d(d);
                return o && t && "function" == typeof t ? c.a.createElement("div", {
                    className: "alternate-sources__dropdown-wrapper"
                }, c.a.createElement(Dt.a, {
                    className: "alternate-sources__dropdown"
                }, b && c.a.createElement(At, {
                    dropdownContent: c.a.createElement(Jt, {
                        createNonBrandedPaperLink: t,
                        paper: n
                    })
                })), c.a.createElement(Wt.a, {
                    feature: at.b.OpenAthensRedirect
                }, c.a.createElement(Gt.a, null, m ? c.a.createElement("div", {
                    className: "institutional-banner"
                }, c.a.createElement("span", {
                    className: "banner-new-label"
                }, Object(F.c)(e => e.paper.institutionalAccessBanner.new)), Object(F.c)(e => e.paper.institutionalAccessBanner.institutionalAccess)) : null))) : c.a.createElement("div", {
                    className: "alternate-sources__dropdown-wrapper"
                }, c.a.createElement(Dt.a, {
                    className: "alternate-sources__dropdown"
                }, c.a.createElement(rr, {
                    paper: n
                }), b && c.a.createElement(ar.a, {
                    ariaLabel: Object(F.c)(e => e.paperDetail.morePaperLinks),
                    className: "alternate-sources__dropdown-menu",
                    content: c.a.createElement(Jt, {
                        createNonBrandedPaperLink: t,
                        paper: n
                    }),
                    testId: "alt-paper-links"
                }, c.a.createElement("span", Object(ye.a)(h), c.a.createElement(ue.a, {
                    width: "10",
                    height: "10",
                    icon: "disclosure",
                    className: "icon-down"
                })))), c.a.createElement(Wt.a, {
                    feature: at.b.OpenAthensRedirect
                }, c.a.createElement(Gt.a, null, m ? c.a.createElement("div", {
                    className: "institutional-banner"
                }, c.a.createElement("span", {
                    className: "banner-new-label"
                }, Object(F.c)(e => e.paper.institutionalAccessBanner.new)), Object(F.c)(e => e.paper.institutionalAccessBanner.institutionalAccess)) : null)))
            }), e => {
                const t = Object(Ce.e)(e.authStore, e => ({
                        isLoggedIn: e.hasAuthenticatedUser()
                    })),
                    r = Object(Ce.e)(e.readerVisibilityStore, e => ({
                        readerVisible: e.getVisibility()
                    }));
                return function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = null != arguments[t] ? arguments[t] : {};
                        t % 2 ? nr(Object(r), !0).forEach((function(t) {
                            ir(e, t, r[t])
                        })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : nr(Object(r)).forEach((function(t) {
                            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                        }))
                    }
                    return e
                }({}, t, {}, {
                    envInfo: e.envInfo
                }, {}, r)
            });

            function or() {
                return c.a.createElement("div", {
                    className: "flex-paper-actions__info flex-paper-actions__info--no-link",
                    "data-test-id": "no-paper-link"
                }, c.a.createElement("div", {
                    className: "flex-paper-actions__info__icon"
                }, c.a.createElement(ue.a, {
                    icon: "fa-link-broken",
                    height: "19",
                    width: "14"
                })), c.a.createElement("div", {
                    className: "flex-paper-actions__info__label"
                }, Object(F.c)(e => e.paper.noLinkLabel)))
            }
            var lr = r(389),
                cr = r(383),
                pr = r(62),
                ur = r(539);

            function dr(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function hr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }

            function mr() {
                return (mr = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }
            class br extends l.Component {
                componentDidMount() {
                    Object(ct.a)(pr.a.create(K.a.PaperDetail.Reader.Button.IMPRESSION))
                }
                render() {
                    var e, t;
                    const {
                        paperId: r,
                        readerVisible: a,
                        isPrimary: n
                    } = this.props;
                    return null != a && null !== (e = a.pdfUrl) && void 0 !== e && e.url ? l.createElement(lr.default, {
                        placement: lr.PLACEMENT.BOTTOM,
                        id: "semantic-reader-tooltip",
                        tooltipContent: Object(F.c)(e => e.paperDetail.readerTooltip)
                    }, l.createElement(Tt.a, {
                        to: "READER",
                        params: {
                            paperId: r,
                            pdfSha: (null == a || null === (t = a.pdfSha) || void 0 === t ? void 0 : t.id) || ""
                        },
                        shouldUnderline: !1,
                        "aria-describedby": "semantic-reader-tooltip",
                        "data-test-id": "reader-button"
                    }, l.createElement(Mt.a, mr({
                        tagName: "a",
                        className: w()("reader__button", {
                            "reader__button--primary": !!n
                        }),
                        iconProps: {
                            height: "100%",
                            width: "100%",
                            icon: "semantic-reader",
                            className: "reader__button-icon"
                        },
                        testId: "reader-button",
                        text: Object(F.c)(e => e.paperDetail.viewReaderButtonWithPDFPrefix),
                        onClick: () => Object(ur.c)(r)
                    }, Object(cr.b)())))) : null
                }
            }
            var fr = Object(Ce.c)(br, e => function(e) {
                for (var t = 1; t < arguments.length; t++) {
                    var r = null != arguments[t] ? arguments[t] : {};
                    t % 2 ? dr(Object(r), !0).forEach((function(t) {
                        hr(e, t, r[t])
                    })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : dr(Object(r)).forEach((function(t) {
                        Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                    }))
                }
                return e
            }({}, Object(Ce.e)(e.readerVisibilityStore, e => ({
                readerVisible: e.getVisibility()
            }))));

            function gr(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function yr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Or extends l.PureComponent {
                constructor() {
                    super(...arguments), yr(this, "onClickPaperLink", (e, t) => {
                        if (e.linkType === Vt.a.WILEY || e.linkType === Vt.a.DOI) {
                            const r = Ft.a(e.url);
                            t.preventDefault(), window.open(r, "_blank")
                        }
                    }), yr(this, "prepareAlternateLinkGroups", e => Ft.g(e))
                }
                componentDidMount() {
                    this.trackImpressionsDataToRico()
                }
                componentDidUpdate(e) {
                    const {
                        paper: t
                    } = this.props;
                    t.id !== e.paper.id && this.trackImpressionsDataToRico()
                }
                trackImpressionsDataToRico() {
                    const {
                        paper: e,
                        showPrimaryOnly: t
                    } = this.props, r = this.prepareAlternateLinkGroups(e.alternatePaperLinks), a = r.nonPdf.map(e => e.url).toArray().concat(r.pdf.map(e => e.url).toArray()), n = r.openAccess.map(e => e.url).toArray(), i = {
                        paper: e.id,
                        primaryPaperLinkDisplayed: e.primaryPaperLink ? e.primaryPaperLink.linkType : null,
                        primaryPaperPublisher: e.primaryPaperLink ? e.primaryPaperLink.publisherName : null,
                        alternateSourcesDisplayed: !(t || e.alternatePaperLinks.isEmpty()),
                        listOfAlternateSourcesDisplayed: a,
                        openAccessDisplayed: !r.openAccess.isEmpty(),
                        listofOpenAccessDisplayed: n
                    };
                    Object(ze.a)(K.a.PaperDetail.IMPRESSION, i)
                }
                getAlternate() {
                    const {
                        envInfo: e,
                        paper: t,
                        readerVisible: r,
                        showPrimaryOnly: a
                    } = this.props, n = $t(r, e);
                    return a || t.alternatePaperLinks.isEmpty() ? n ? l.createElement("div", {
                        className: "alternate-sources__dropdown-wrapper"
                    }, l.createElement(Dt.a, {
                        className: "alternate-sources__dropdown"
                    }, l.createElement(At, {
                        dropdownContent: this.createNonBrandedPaperLink()
                    }))) : null : l.createElement(sr, {
                        createNonBrandedPaperLink: () => this.createNonBrandedPaperLink(),
                        paper: t
                    })
                }
                createNonBrandedPaperLink() {
                    const {
                        entitlement: e,
                        paper: t
                    } = this.props, {
                        primaryPaperLink: r
                    } = this.props.paper, a = e ? Object(Kt.b)({
                        url: e.url,
                        linkType: "entitlement",
                        publisherName: "none"
                    }) : r;
                    if (!a) return null;
                    return Ft.e(a) ? null : l.createElement("ul", null, l.createElement("li", null, l.createElement(Yt, {
                        link: a,
                        paperId: t.id,
                        isPrimary: !1
                    })))
                }
                renderViewPaperAsPrimary(e) {
                    let {
                        hasPrimaryLink: t,
                        showReaderButton: r,
                        primaryButtonLink: a
                    } = e;
                    const {
                        entitlement: n,
                        paper: i
                    } = this.props;
                    return l.createElement(l.Fragment, null, t && l.createElement(Dt.a, {
                        className: "alternate-sources__paperlink-wrapper"
                    }, l.createElement(Yt, {
                        className: "flex-paper-actions__button flex-paper-actions__button--primary",
                        link: a,
                        paperId: i.id,
                        isPrimary: !0,
                        onClick: this.onClickPaperLink,
                        entitlement: n
                    })), r && l.createElement(fr, {
                        paperId: i.id,
                        isPrimary: !0
                    }))
                }
                renderReaderAsPrimary(e) {
                    let {
                        hasPrimaryLink: t,
                        isBranded: r,
                        showReaderButton: a,
                        primaryButtonLink: n
                    } = e;
                    const {
                        entitlement: i,
                        paper: s
                    } = this.props;
                    return l.createElement(l.Fragment, null, t && r && l.createElement(Dt.a, {
                        className: "alternate-sources__paperlink-wrapper"
                    }, l.createElement(Yt, {
                        className: "flex-paper-actions__button flex-paper-actions__button--primary",
                        link: n,
                        paperId: s.id,
                        isPrimary: !0,
                        onClick: this.onClickPaperLink,
                        entitlement: i
                    })), a && l.createElement(fr, {
                        paperId: s.id,
                        isPrimary: !0
                    }))
                }
                render() {
                    const {
                        entitlement: e,
                        envInfo: t,
                        paper: {
                            primaryPaperLink: r
                        },
                        readerVisible: a
                    } = this.props, n = $t(a, t), i = e ? Object(Kt.b)({
                        url: e.url,
                        linkType: "entitlement",
                        publisherName: "none"
                    }) : r, s = Ft.e(i), o = this.getAlternate(), c = !!i, p = c || !!o;
                    return l.createElement(te.a, {
                        className: w()("flex-paper-actions__item-container", {
                            reader__container: n
                        }),
                        wrap: !1
                    }, !p && l.createElement(Dt.a, {
                        className: "alternate-sources__paperlink-wrapper"
                    }, l.createElement(or, null)), !!i && (n && !s ? this.renderReaderAsPrimary({
                        hasPrimaryLink: c,
                        isBranded: s,
                        showReaderButton: n,
                        primaryButtonLink: i
                    }) : this.renderViewPaperAsPrimary({
                        hasPrimaryLink: c,
                        showReaderButton: n,
                        primaryButtonLink: i
                    })), o)
                }
            }
            var Er = Object(Ce.c)(Or, e => {
                    const t = Object(Ce.e)(e.authStore, e => ({
                            isLoggedIn: e.hasAuthenticatedUser()
                        })),
                        r = Object(Ce.e)(e.readerVisibilityStore, e => ({
                            readerVisible: e.getVisibility()
                        }));
                    return function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? gr(Object(r), !0).forEach((function(t) {
                                yr(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : gr(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, t, {}, {
                        envInfo: e.envInfo
                    }, {}, r)
                }),
                vr = r(19);
            class Pr extends l.PureComponent {
                render() {
                    const {
                        onClick: e,
                        className: t
                    } = this.props, r = Object(F.c)(e => e.paper.action.cite), a = w()("cite-button", t);
                    return c.a.createElement(Mt.a, {
                        iconProps: {
                            icon: "fa-quote",
                            width: "11",
                            height: "11"
                        },
                        target: "_blank",
                        onClick: e,
                        className: a,
                        testId: "cite-link",
                        text: r
                    })
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(Pr, "propTypes", {
                paper: O.a.instanceOf(vr.d).isRequired,
                className: O.a.string,
                onClick: O.a.func.isRequired
            });
            var Sr = r(184),
                wr = r(74),
                _r = r(98),
                Cr = r(388),
                xr = r(411),
                jr = r(132),
                Tr = r(9),
                kr = r(134),
                Ir = r(135);

            function Nr(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Dr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Rr extends l.PureComponent {
                constructor() {
                    super(...arguments), Dr(this, "onClickOpenOrganizePapersShelf", e => {
                        e.stopPropagation(), e.preventDefault(), this.openOrganizePapersShelf()
                    }), Dr(this, "openOrganizePapersShelf", () => {
                        const {
                            dispatcher: e
                        } = this.context, {
                            paper: t
                        } = this.props, r = Object(Cr.d)({
                            paperId: t.id,
                            paperTitle: Object(Sr.c)(t)
                        });
                        e.dispatch(r)
                    }), Dr(this, "fetchLibraryFolders", () => {
                        const {
                            api: e,
                            libraryFolderStore: t
                        } = this.context;
                        return t.isUninitialized() ? e.getLibraryFolders() : Promise.resolve()
                    }), Dr(this, "handleSaveToLibraryWithShelf", () => {
                        const {
                            paper: e
                        } = this.props, {
                            api: t,
                            messageStore: r,
                            dispatcher: a
                        } = this.context;
                        this.fetchLibraryFolders().then(() => {
                            const n = Object(xr.a)({
                                paperId: e.id,
                                paperTitle: Object(Sr.c)(e),
                                sourceType: wr.c
                            });
                            t.createLibraryEntryBulk(n).catch(e => {
                                r.addError(Object(F.c)(e => e.library.saveToLibraryShelf.errorMessage)), Tr.a.error(e)
                            }), a.dispatch(Object(Cr.f)({
                                paper: e
                            }))
                        }).catch(t => {
                            Object(We.default)("library", `failed to open Save To Library shelf for paperId="${e.id}"]`, t);
                            const a = Object(F.c)(e => e.library.message.error.header),
                                n = Object(F.c)(e => e.library.message.error.body);
                            r.addError(n, a)
                        })
                    }), Dr(this, "onClickSaveToLibrary", () => {
                        const {
                            onSaveClick: e,
                            trackLibraryClick: t
                        } = this.props, {
                            authStore: r,
                            dispatcher: a
                        } = this.context;
                        "function" == typeof e && e(), r.ensureLogin({
                            dispatcher: a,
                            location: _r.g.library
                        }).then(async () => {
                            t && t("READING_LIST", {
                                action: "save"
                            }), await this.fetchLibraryFolders(), this.getStateFromLibraryFolderStore().isInLibrary ? this.openOrganizePapersShelf() : this.handleSaveToLibraryWithShelf()
                        }, e => {
                            Tr.a.warn(e)
                        })
                    }), Dr(this, "buildGenericSaveButton", () => l.createElement(N.default, {
                        className: this.props.className,
                        label: this.props.buttonText || "Add to Library",
                        onClick: this.onClickSaveToLibrary
                    }));
                    const {
                        libraryFolderStore: e
                    } = this.context;
                    this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? Nr(Object(r), !0).forEach((function(t) {
                                Dr(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Nr(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromLibraryFolderStore()), e.registerComponent(this, () => {
                        this.setState(this.getStateFromLibraryFolderStore())
                    })
                }
                componentWillUnmount() {
                    const {
                        messageStore: e
                    } = this.context;
                    e.clearMessages()
                }
                getStateFromLibraryFolderStore() {
                    const {
                        libraryFolderStore: e
                    } = this.context, {
                        paper: t
                    } = this.props;
                    return {
                        isInLibrary: e.isPaperInLibrary(t.id)
                    }
                }
                render() {
                    const {
                        isInLibrary: e
                    } = this.state, t = e ? Object(F.c)(e => e.paper.action.inLibraryResponsiveText) : Object(F.c)(e => e.paper.action.saveShort), r = this.props.buttonText ? this.props.buttonText : Object(F.c)(e => e.paper.action.saveShort), a = this.props.saveButtonIcon ? this.props.saveButtonIcon : "fa-bookmark", n = this.props.saveButtonIcon ? "18" : "11", i = l.createElement(Mt.a, {
                        responsiveText: t,
                        className: this.props.className,
                        testId: "paper-action-save",
                        onClick: this.onClickSaveToLibrary,
                        iconProps: {
                            icon: a,
                            width: n,
                            height: n
                        },
                        text: r
                    }), s = l.createElement(Tt.a, {
                        to: "LIBRARY_ALL_ENTRIES",
                        shouldUnderline: !1
                    }, l.createElement(Mt.a, {
                        responsiveText: t,
                        onClick: this.onClickOpenOrganizePapersShelf,
                        className: this.props.className,
                        testId: "paper-action-view-library",
                        iconProps: {
                            icon: "fa-bookmark",
                            width: n,
                            height: n
                        },
                        text: Object(F.c)(e => e.paper.action.goToLibrary)
                    })), o = this.props.generic ? this.buildGenericSaveButton() : i;
                    return e ? s : o
                }
            }
            Dr(Rr, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired,
                authStore: O.a.instanceOf(jr.a).isRequired,
                dispatcher: O.a.instanceOf(g.a).isRequired,
                libraryFolderStore: O.a.instanceOf(kr.a).isRequired,
                messageStore: O.a.instanceOf(Ir.a).isRequired
            });
            var Lr = r(97),
                Ar = r(136),
                Fr = r(39),
                Mr = r(109);

            function qr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Br extends l.PureComponent {
                constructor() {
                    super(...arguments), qr(this, "showCiteModal", () => {
                        Object(ct.a)(xt.a.create(K.a.PaperDetail.Header.Action.CITE)), this.context.dispatcher.dispatch(Object(Mr.b)({
                            id: Fr.b.CITE,
                            data: this.props.paper
                        }))
                    })
                }
                render() {
                    const {
                        paper: e,
                        trackActionClick: t,
                        entitlement: r
                    } = this.props;
                    return c.a.createElement(te.a, {
                        wrap: !0,
                        className: "flex-paper-actions__group"
                    }, c.a.createElement(Dt.a, {
                        className: "alternate-sources"
                    }, c.a.createElement(Er, {
                        paper: e,
                        entitlement: r
                    })), c.a.createElement(Dt.a, null, c.a.createElement(te.a, {
                        className: "flex-paper-actions__item-container"
                    }, c.a.createElement(Dt.a, null, c.a.createElement(ye.b, {
                        heapProps: {
                            id: Oe.m
                        }
                    }, c.a.createElement(Rr, {
                        buttonText: Object(F.c)(e => e.paper.action.save),
                        className: "flex-paper-actions__button flex-paper-actions__button--responsive-secondary",
                        paper: e,
                        trackLibraryClick: t
                    }))), c.a.createElement(Dt.a, null, c.a.createElement(ye.b, {
                        heapProps: {
                            id: Oe.k
                        }
                    }, c.a.createElement(Rt.a, {
                        className: "flex-paper-actions__button flex-paper-actions__button--responsive-secondary",
                        queryType: Lr.c.PAPER_CITATION,
                        queryValue: e.id,
                        displayValue: e.title.text,
                        subLocation: _r.g.pdpAlert
                    }))), c.a.createElement(Dt.a, {
                        className: "flex-paper-actions__item--hidden-when-narrow"
                    }, c.a.createElement(ye.b, {
                        heapProps: {
                            id: Oe.l
                        }
                    }, c.a.createElement(Pr, {
                        className: "flex-paper-actions__button flex-paper-actions__button--responsive-secondary",
                        paper: e,
                        onClick: this.showCiteModal
                    }))))))
                }
            }
            qr(Br, "propTypes", {
                paper: O.a.object.isRequired,
                trackActionClick: O.a.func.isRequired,
                isPdp: O.a.bool,
                entitlement: O.a.object
            }), qr(Br, "contextTypes", {
                alertsStore: O.a.instanceOf(Ar.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                dispatcher: O.a.instanceOf(g.a).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var Vr = r(382),
                Hr = r(183),
                Qr = r.n(Hr);

            function Ur(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class zr extends l.PureComponent {
                constructor() {
                    super(...arguments), Ur(this, "portalNode", void 0), this.portalNode = null
                }
                componentWillUnmount() {
                    this.destroyPortalNode()
                }
                createPortalNode() {
                    this.destroyPortalNode();
                    const e = document.createElement("div");
                    return e.className = "preview-box__portal", document.body && document.body.appendChild(e), this.portalNode = e, e
                }
                destroyPortalNode() {
                    const e = this.portalNode;
                    if (!e) return;
                    const {
                        parentNode: t
                    } = e;
                    t && (t.removeChild(e), this.portalNode = null)
                }
                render() {
                    const {
                        topPx: e,
                        heightPx: t,
                        leftPx: r,
                        position: a,
                        widthPx: n,
                        theme: i,
                        children: s,
                        className: o,
                        isVisible: c,
                        usePortal: p
                    } = this.props, u = l.createElement("div", {
                        className: w()({
                            "preview-box__anchor-point": !0,
                            "preview-box__is-visible": c
                        }),
                        style: {
                            top: e + "px",
                            left: r + "px"
                        }
                    }, l.createElement("div", {
                        className: w()({
                            "preview-box": !0,
                            ["preview-box__theme-" + (i || "")]: !!i,
                            ["preview-box__position-" + a]: !0,
                            [o || ""]: !!o
                        }),
                        style: {
                            height: "number" == typeof t && t >= 0 ? t + "px" : void 0,
                            width: "number" == typeof n && n >= 0 ? n + "px" : void 0
                        }
                    }, l.createElement("div", {
                        className: "preview-box__content"
                    }, s)));
                    if (p) {
                        const e = this.portalNode || this.createPortalNode();
                        return Qr.a.createPortal(u, e)
                    }
                    return u
                }
            }

            function Yr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            Ur(zr, "defaultProps", {
                isVisible: !0,
                usePortal: !0
            });
            class Wr extends l.PureComponent {
                constructor() {
                    super(...arguments), Yr(this, "onMouseEnter", () => {
                        clearTimeout(this.showTimer), clearTimeout(this.hideTimer), this.showTimer = setTimeout(() => {
                            this.showPreview()
                        }, this.props.showDelayMs)
                    }), Yr(this, "onMouseLeave", () => {
                        clearTimeout(this.hideTimer), clearTimeout(this.showTimer), this.hideTimer = setTimeout(() => {
                            this.hidePreview()
                        }, this.props.hideDelayMs)
                    }), Yr(this, "getRef", e => {
                        this.ref = e
                    }), this.state = {
                        boxPosition: {},
                        isPreviewShown: !1
                    }, this.ref = null, this.hideTimer = null, this.showTimer = null
                }
                showPreview() {
                    "function" == typeof this.props.onShow && this.props.onShow(), this.setState(() => ({
                        boxPosition: this.calculateBoxPosition(),
                        isPreviewShown: !0
                    }))
                }
                hidePreview() {
                    this.setState(() => ({
                        isPreviewShown: !1
                    }))
                }
                calculateBoxPosition() {
                    const e = He.a.getViewportSize();
                    if (!this.ref || !e || !window) return {
                        position: "top-left",
                        topPx: 0,
                        leftPx: 0
                    };
                    const t = this.ref.getBoundingClientRect(),
                        r = t.y + t.height / 2 + window.scrollY < window.scrollY + e.height / 2 ? "bottom" : "top",
                        a = t.x + t.width / 2 + window.scrollX,
                        n = window.scrollX + e.width / 3,
                        i = window.scrollX + e.width / 3 * 2;
                    return {
                        position: `${r}-${a<n?"right":i<a?"left":"center"}`,
                        topPx: ("top" === r ? t.top : t.top + t.height) + window.scrollY,
                        leftPx: t.left + t.width / 2 + window.scrollX
                    }
                }
                renderPreviewBox() {
                    const {
                        previewBox: e
                    } = this.props, {
                        boxPosition: t
                    } = this.state;
                    return !c.a.Children.toArray(e).find(e => e.type !== zr) ? c.a.cloneElement(e, t) : c.a.createElement(zr, t, e)
                }
                render() {
                    const {
                        isPreviewShown: e
                    } = this.state;
                    return c.a.createElement("span", {
                        className: "preview-box__target",
                        ref: this.getRef,
                        onMouseEnter: this.onMouseEnter,
                        onMouseLeave: this.onMouseLeave
                    }, this.props.children, e && this.renderPreviewBox())
                }
            }
            Yr(Wr, "propTypes", {
                previewBox: O.a.node.isRequired,
                children: O.a.node.isRequired,
                onShow: O.a.func,
                hideDelayMs: O.a.number
            }), Yr(Wr, "defaultProps", {
                hideDelayMs: 200,
                showDelayMs: 150
            });
            var Gr = r(395),
                Kr = r(540);

            function $r(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Xr(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Zr extends l.PureComponent {
                constructor() {
                    super(...arguments), Xr(this, "handleHover", () => {
                        const e = Kr.a.create(K.a.PaperDetail.Abstract.KEY_RESULT_HOVER, function(e) {
                            for (var t = 1; t < arguments.length; t++) {
                                var r = null != arguments[t] ? arguments[t] : {};
                                t % 2 ? $r(Object(r), !0).forEach((function(t) {
                                    Xr(e, t, r[t])
                                })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : $r(Object(r)).forEach((function(t) {
                                    Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                                }))
                            }
                            return e
                        }({}, this.context.heapPropsChain, {
                            paperId: this.props.paperDetail.paper.id
                        }));
                        Object(ct.a)(e)
                    }), Xr(this, "truncateContent", e => {
                        const t = e.trim().split(" "),
                            r = Object(F.c)(e => e.paperDetail.abstract.highlight.paragraphEllipsis);
                        return r + t.slice(0, 49).join(" ") + r
                    }), Xr(this, "getPDFLink", () => {
                        const {
                            relatedPages: e,
                            paperDetail: t
                        } = this.props, r = t.paper.primaryPaperLink;
                        if (r && Ft.f(r.url) && e && e.length > 0) {
                            return `${r.url}#page=${e[0]}`
                        }
                        return null
                    })
                }
                renderPreviewBox() {
                    const {
                        content: e
                    } = this.props, t = this.getPDFLink();
                    return l.createElement("div", {
                        className: "entity-preview-box abstract-result__paragraph-preview",
                        "data-test-id": "entity-preview-box"
                    }, l.createElement("div", {
                        className: "entity-preview-box-hover-text abstract-result__related-paragraph__box"
                    }, l.createElement("div", {
                        className: "abstract-result__related-paragraph__title"
                    }, Object(F.c)(e => e.paperDetail.abstract.highlight.paragraphLabel)), l.createElement("div", {
                        className: "abstract-result__related-paragraph__text"
                    }, this.truncateContent(e)), t && l.createElement(Gr.a, {
                        "data-heap-id": "paper-abstract-highlight-related-pdf-link-click",
                        href: t,
                        className: "abstract-result__related-paragraph__pdf-link"
                    }, l.createElement("div", {
                        className: "abstract-result__related-paragraph__pdf-link__text"
                    }, Object(F.c)(e => e.paperDetail.abstract.highlight.pdfLinkLabel), l.createElement(ue.a, {
                        icon: "arrow-lite",
                        className: "abstract-result__related-paragraph__pdf-link__arrow",
                        height: "12",
                        width: "12"
                    })))))
                }
                render() {
                    return l.createElement(Wr, {
                        previewBox: this.renderPreviewBox(),
                        onShow: this.handleHover,
                        children: this.props.previewTarget
                    })
                }
            }
            const Jr = "result_label",
                ea = "method_label";
            var ta = r(564);

            function ra() {
                return (ra = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function aa(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            const na = Et.b.Map({
                [Jr]: "key-result__highlight",
                [ea]: "key-method__highlight"
            });
            class ia extends l.PureComponent {
                constructor() {
                    super(...arguments), aa(this, "getKeyFlag", e => {
                        const {
                            isTruncated: t
                        } = this.props;
                        return t ? e[0].label === ea ? Object(F.c)(e => e.paperDetail.abstract.highlight.keyMethodSingle) : Object(F.c)(e => e.paperDetail.abstract.highlight.keyResultSingle) : null
                    }), this.state = {
                        isTruncated: this.props.isTruncated
                    }
                }
                componentDidUpdate(e) {
                    e.isTruncated !== this.props.isTruncated && this.updateTruncatedState()
                }
                updateTruncatedState() {
                    this.setState({
                        isTruncated: this.props.isTruncated
                    })
                }
                renderHighlightedAbstract(e) {
                    const {
                        paperDetail: t,
                        offset: r,
                        skipperAbstractData: a,
                        sentenceTypes: n
                    } = this.props;
                    let i = 0;
                    const s = [],
                        o = a.sentences.filter(e => e.label && n.has(e.label));
                    for (let a = 0; a < o.length; ++a) {
                        const n = o[a],
                            c = n.startIndex - r;
                        let p = n.endIndex - r;
                        if (0 == a && p > e.length && (p = e.length), 0 != c && (s.push(e.slice(i, c)), c > e.length)) return s;
                        if (!(c < e.length && p <= e.length)) return s;
                        {
                            const r = this.getKeyFlag(o);
                            0 == a && r && s.push(l.createElement("span", {
                                className: "paper-detail__abstract__highlighted__key-highlight",
                                key: a
                            }, r));
                            const u = l.createElement("mark", {
                                className: na.get(n.label),
                                key: a + .5
                            }, e.slice(c, p));
                            s.push(n.paragraph ? l.createElement(Zr, {
                                key: a + .75,
                                relatedPages: n.relatedPages,
                                paperDetail: t,
                                content: n.paragraph,
                                previewTarget: u
                            }) : u), i = p
                        }
                    }
                    return i < e.length && s.push(e.slice(i, e.length)), s
                }
                renderToggle() {
                    const {
                        isTruncated: e
                    } = this.state, {
                        withEllipsis: t,
                        isBeginningTruncated: r
                    } = this.props, a = e || r ? "more" : "less", n = w()("text-truncator__abstract__toggle", "link-button", a), i = e ? this.props.extendActionText : this.props.truncateActionText, s = e ? this.props.extendAriaLabel : this.props.truncateAriaLabel, o = "more" === a ? "square-plus" : "square-minus", c = l.createElement("button", ra({
                        className: n,
                        "aria-label": s,
                        onClick: this.props.onClick,
                        "data-test-id": "text-truncator-toggle"
                    }, Object(ta.a)()), l.createElement("span", {
                        className: "paper-detail__abstract__toggle__label"
                    }, (t || !e || !r) && i && l.createElement(ue.a, {
                        className: "paper-detail__abstract__toggle__icon",
                        icon: o,
                        height: "12",
                        width: "12"
                    }), i));
                    return t || !e ? c : l.createElement("span", null, " [", c, "] ")
                }
                render() {
                    const {
                        limit: e,
                        text: t,
                        withEllipsis: r
                    } = this.props, a = ne.m(t, e, r);
                    return l.createElement("span", {
                        className: w()("text-truncator", "abstract__text text--preline")
                    }, this.renderHighlightedAbstract(this.state.isTruncated ? a : t), this.renderToggle())
                }
            }
            aa(ia, "defaultProps", {
                limit: 360,
                isTruncated: !0,
                isBeginningTruncated: !1,
                extendActionText: Object(F.c)(e => e.textTruncator.default.extendLabel),
                extendAriaLabel: Object(F.c)(e => e.textTruncator.default.extendAriaLabel),
                truncateActionText: Object(F.c)(e => e.textTruncator.default.truncateLabel),
                truncateAriaLabel: Object(F.c)(e => e.textTruncator.default.truncateAriaLabel),
                withEllipsis: !0,
                offset: 0,
                sentenceTypes: new Set
            });
            var sa = r(3),
                oa = r.n(sa);

            function la(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function ca(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class pa extends l.PureComponent {
                constructor(e) {
                    for (var t = arguments.length, r = new Array(t > 1 ? t - 1 : 0), a = 1; a < t; a++) r[a - 1] = arguments[a];
                    super(e, ...r), ca(this, "trackAbstractHighlightEvent", e => {
                        const t = xt.a.create(e, function(e) {
                            for (var t = 1; t < arguments.length; t++) {
                                var r = null != arguments[t] ? arguments[t] : {};
                                t % 2 ? la(Object(r), !0).forEach((function(t) {
                                    ca(e, t, r[t])
                                })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : la(Object(r)).forEach((function(t) {
                                    Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                                }))
                            }
                            return e
                        }({}, this.context.heapPropsChain, {
                            paperId: this.props.paperDetail.paper.id
                        }));
                        Object(ct.a)(t)
                    }), ca(this, "openFullAbstract", () => {
                        let {
                            isBeginningTruncated: e
                        } = this.state;
                        this.setState(t => ((e && t.isTruncated || !e && !t.isTruncated) && (e = !e), {
                            isTruncated: !t.isTruncated,
                            isBeginningTruncated: e
                        }))
                    }), ca(this, "onToggleClick", () => {
                        this.openFullAbstract(), this.state.isTruncated ? this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.TOGGLE_CLOSE) : this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.TOGGLE_OPEN)
                    }), ca(this, "onBeginningToggleClick", () => {
                        this.setState(e => ({
                            isBeginningTruncated: !e.isBeginningTruncated
                        }), () => this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.TOGGLE_BEGINNING))
                    }), ca(this, "onClickMethods", () => {
                        const {
                            isMethodsHighlighted: e,
                            sentenceTypes: t,
                            isTruncated: r
                        } = this.state;
                        r && this.openFullAbstract(), e ? t.delete(ea) : t.add(ea), this.setState({
                            isMethodsHighlighted: !e,
                            sentenceTypes: new Set(t)
                        }, () => {
                            this.state.isMethodsHighlighted ? this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.HIGHLIGHT_TOGGLE_METHODS_ON) : this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.HIGHLIGHT_TOGGLE_METHODS_OFF)
                        })
                    }), ca(this, "onClickResults", () => {
                        const {
                            isResultsHighlighted: e,
                            sentenceTypes: t,
                            isTruncated: r
                        } = this.state;
                        r && this.openFullAbstract(), e ? t.delete(Jr) : t.add(Jr), this.setState({
                            isResultsHighlighted: !e,
                            sentenceTypes: new Set(t)
                        }, () => {
                            this.state.isResultsHighlighted ? this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.HIGHLIGHT_TOGGLE_RESULTS_ON) : this.trackAbstractHighlightEvent(K.a.PaperDetail.Abstract.HIGHLIGHT_TOGGLE_RESULTS_OFF)
                        })
                    }), ca(this, "renderHighlightToggle", e => {
                        const {
                            isMethodsHighlighted: t,
                            isResultsHighlighted: r
                        } = this.state, a = e.find(e => e.label === ea), n = e.find(e => e.label === Jr);
                        return l.createElement("div", {
                            className: "paper-detail__abstract__highlight__toggle"
                        }, l.createElement("span", {
                            className: "paper-detail__abstract__highlight__toggle__buttons__label"
                        }, l.createElement(ue.a, {
                            icon: "highlighter",
                            width: "14",
                            height: "14",
                            className: "paper-detail__abstract__highlight__toggle__buttons__label__icon"
                        }), l.createElement("span", {
                            className: "paper-detail__abstract__highlight__toggle__buttons__label__text"
                        }, Object(F.c)(e => e.paperDetail.abstract.highlight.highlightToggleLabel))), l.createElement("span", {
                            className: "paper-detail__abstract__highlight__toggle__buttons"
                        }, a && l.createElement(N.default, {
                            className: w()({
                                abstract__highlight__toggle__button: !0,
                                abstract__highlight__toggle__button__method__on: t
                            }),
                            label: Object(F.c)(e => e.paperDetail.abstract.highlight.methodsToggleLabel),
                            onClick: this.onClickMethods
                        }), n && l.createElement(N.default, {
                            className: w()({
                                abstract__highlight__toggle__button: !0,
                                abstract__highlight__toggle__button__result__on: r
                            }),
                            label: Object(F.c)(e => e.paperDetail.abstract.highlight.resultsToggleLabel),
                            onClick: this.onClickResults
                        })))
                    });
                    const {
                        paperDetail: n
                    } = this.props, i = n.skipperExperiments.dataFor("paper-abstract-highlight-v2");
                    oa()(i, "experiment data must be present");
                    const s = i.sentences.find(e => e.label === Jr || e.label === ea),
                        o = s && s.label;
                    this.state = {
                        skipperAbstractData: i,
                        isTruncated: !0,
                        isBeginningTruncated: !0,
                        sentenceTypes: new Set([o]),
                        isResultsHighlighted: o === Jr,
                        isMethodsHighlighted: o === ea
                    }
                }
                render() {
                    const {
                        skipperAbstractData: e,
                        isTruncated: t,
                        isBeginningTruncated: r,
                        sentenceTypes: a
                    } = this.state, {
                        isMobile: n,
                        paperDetail: i
                    } = this.props, s = e.sentences.find(e => e.label && a.has(e.label)), o = s ? s.startIndex : 0, c = !(!o || 0 == o), p = e.firstSentenceEndIndex, u = n ? 200 : 500, d = c ? u - p : u;
                    return l.createElement("div", {
                        className: "fresh-paper-detail-page__abstract",
                        "data-test-id": "abstract-text"
                    }, l.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: "Abstract"
                    }), !t && this.renderHighlightToggle(e.sentences), l.createElement("div", {
                        className: "paper-detail__abstract__highlighted"
                    }, c && l.createElement(ia, {
                        disableClickStyle: !0,
                        onClick: this.onBeginningToggleClick,
                        isTruncated: r,
                        isBeginningTruncated: r,
                        skipperAbstractData: e,
                        text: e.abstract.substring(0, o),
                        limit: p,
                        extendActionText: Object(F.c)(e => e.paperDetail.abstract.highlight.beginningToggle),
                        truncateActionText: Object(F.c)(e => e.paperDetail.abstract.highlight.noToggleExpansion),
                        withEllipsis: !1,
                        paperDetail: i
                    }), l.createElement(ia, {
                        isTruncated: t,
                        isBeginningTruncated: r,
                        onClick: this.onToggleClick,
                        skipperAbstractData: e,
                        text: e.abstract.substring(o, e.abstract.length),
                        limit: d,
                        offset: o,
                        extendActionText: Object(F.c)(e => e.paperDetail.abstract.highlight.toggleExpand),
                        truncateActionText: Object(F.c)(e => e.paperDetail.abstract.highlight.toggleCollapse),
                        paperDetail: i,
                        sentenceTypes: a
                    })))
                }
            }
            var ua = r(622),
                da = r(520);
            class ha extends l.PureComponent {
                handleClick(e, t) {
                    const r = this.props.onClick;
                    "function" == typeof r && r(e, t, this.props.paper.id)
                }
                renderAuthorsListItem() {
                    const {
                        authors: e
                    } = this.props.paper;
                    return e.isEmpty() ? null : l.createElement(da.a, {
                        authors: e,
                        onAuthorClick: e => {
                            this.handleClick("AUTHOR", e)
                        },
                        max: this.props.maxAuthors,
                        shouldLinkToAHP: !this.props.disableAuthorLinks
                    })
                }
                render() {
                    return l.createElement("span", {
                        className: "paper-meta",
                        "data-test-id": "paper-meta-subhead"
                    }, this.renderAuthorsListItem())
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(ha, "contextTypes", {
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var ma = r(495);

            function ba() {
                return (ba = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function fa(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function ga(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class ya extends c.a.Component {
                constructor() {
                    super(...arguments), ga(this, "onResize", void 0), ga(this, "onScroll", void 0), ga(this, "navRef", void 0), ga(this, "placeholderRef", void 0), ga(this, "scrollRef", void 0), ga(this, "setNavRef", e => {
                        this.navRef = e
                    }), ga(this, "setPlaceholderRef", e => {
                        this.placeholderRef = e
                    }), ga(this, "setScrollRef", e => {
                        this.scrollRef = e
                    }), ga(this, "onViewportChange", () => {
                        const e = Object(pt.b)(document);
                        this.context.dispatcher.dispatchEventually({
                            actionType: f.a.actions.PAPER_NAV_DOM_DIRTY,
                            viewportRect: e
                        });
                        const t = this.shouldBeAttached(e);
                        null !== t && t !== this.state.isAttached ? this.setState({
                            isAttached: t
                        }, () => {
                            t && this.updateHorizontalScrollPos()
                        }) : this.state.isAttached && this.updateHorizontalScrollPos()
                    }), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? fa(Object(r), !0).forEach((function(t) {
                                ga(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : fa(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({
                        isAttached: !1,
                        scrollLeftPx: 0
                    }, this.getStateFromPaperNavStore()), this.context.paperNavStore.registerComponent(this, () => {
                        this.setState(this.getStateFromPaperNavStore())
                    }), this.onResize = Object(rt.a)(this.onViewportChange, 250, {
                        trailing: !0
                    }), this.onScroll = Object(rt.a)(this.onViewportChange, 100, {
                        leading: !0,
                        maxWait: 100,
                        trailing: !0
                    })
                }
                getStateFromPaperNavStore() {
                    const {
                        paperNavStore: e
                    } = this.context;
                    return {
                        isPending: e.isPending(),
                        navItems: e.getSortedNavItems()
                    }
                }
                componentDidMount() {
                    He.a.listenForScroll(this.onScroll), window.addEventListener("resize", this.onResize), this.onViewportChange()
                }
                componentWillUnmount() {
                    He.a.stopListeningForScroll(this.onScroll), window.removeEventListener("resize", this.onResize)
                }
                shouldComponentUpdate(e) {
                    if (this.state.isAttached !== e.isAttached || this.state.isPending !== e.isPending || this.state.navItems.size !== e.navItems.size) return !0;
                    for (let t = 0; t < e.navItems.size; t++) {
                        const r = this.state.navItems.get(t),
                            a = e.navItems.get(t);
                        if (r.isActive !== a.isActive || r.isSolitary !== a.isSolitary || r.isProminent !== a.isProminent || r.isContained !== a.isContained || r.isVisible !== a.isVisible || r.id !== a.id || r.label !== a.label) return !0
                    }
                    return !1
                }
                shouldBeAttached(e) {
                    const t = this.placeholderRef;
                    if (!t) return null;
                    return t.offsetTop <= e.top + 64
                }
                updateHorizontalScrollPos() {
                    const e = this.scrollRef;
                    if (!e) return;
                    const t = this.state.navItems.filter(e => e.isProminent).map(e => e.id).map(t => e.querySelector(`[data-nav-id="${t}"]`)).map(e => e && e.offsetLeft).first();
                    "number" == typeof t && (He.a.hasNativeSmoothScrollSupport() ? e.scrollTo({
                        left: t,
                        behavior: "smooth"
                    }) : e.scrollLeft = t)
                }
                renderNav() {
                    const {
                        navItems: e,
                        isPending: t,
                        isAttached: r
                    } = this.state;
                    return e.isEmpty() ? null : c.a.createElement("nav", {
                        className: w()({
                            "paper-nav__nav": !0,
                            "paper-nav__is-attached": r
                        }),
                        ref: this.setNavRef
                    }, !t && !e.isEmpty() && c.a.createElement(c.a.Fragment, null, c.a.createElement("div", {
                        className: "paper-nav__scrollable",
                        ref: this.setScrollRef
                    }, c.a.createElement("ul", {
                        className: "paper-nav__nav-list"
                    }, e.map(e => c.a.createElement("li", {
                        key: e.id,
                        className: w()({
                            "paper-nav__nav-item": !0,
                            "paper-nav__is-active": e.isActive,
                            "paper-nav__is-solitary": e.isSolitary,
                            "paper-nav__is-prominent": e.isProminent,
                            "paper-nav__is-contained": e.isContained,
                            "paper-nav__is-visible": e.isVisible
                        })
                    }, c.a.createElement(ma.a, ba({
                        navId: e.id,
                        className: "paper-nav__nav-link"
                    }, Object(ye.a)({
                        id: Oe.j,
                        nav: e.id
                    })), c.a.createElement("span", {
                        className: "paper-nav__nav-label"
                    }, e.label))))))))
                }
                render() {
                    const {
                        isAttached: e
                    } = this.state, t = this.navRef;
                    if (t && !this.state.isAttached) {
                        const e = t.clientHeight;
                        if ("undefined" != typeof document) {
                            const t = document.documentElement.style;
                            e !== +t.getPropertyValue("--nav-ref-placeholder-height") && t.setProperty("--nav-ref-placeholder-height", e + "px")
                        }
                    }
                    return c.a.createElement("div", {
                        className: w()({
                            "paper-nav": !0,
                            "paper-nav__is-attached": e,
                            "paper-nav__is-mobile": this.context.envInfo.isMobile
                        })
                    }, c.a.createElement("div", {
                        className: "paper-nav__placeholder",
                        ref: this.setPlaceholderRef
                    }, this.renderNav()))
                }
            }
            ga(ya, "contextTypes", {
                dispatcher: O.a.instanceOf(g.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                paperNavStore: O.a.instanceOf(Ue.a).isRequired
            });
            class Oa extends l.PureComponent {
                render() {
                    const {
                        children: e
                    } = this.props;
                    return l.createElement("div", {
                        className: "card-content-aside"
                    }, e)
                }
            }
            class Ea extends l.PureComponent {
                render() {
                    const {
                        type: e,
                        isFiltered: t
                    } = this.props;
                    if ("citingPapers" === e && t) return c.a.createElement("div", {
                        className: "empty-pdp-box"
                    }, Object(F.c)(e => e.citations.emptyFiltered.text));
                    const r = "citingPapers" === e ? Object(F.c)(e => e.citations.empty.text) : Object(F.c)(e => e.references.empty.text);
                    return c.a.createElement("div", {
                        className: "empty-pdp-box"
                    }, r)
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(Ea, "propTypes", {
                type: O.a.string.isRequired,
                isFiltered: O.a.boolean
            });
            class va extends ue.a {}! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(va, "defaultProps", {
                icon: "help",
                width: "15",
                height: "15",
                className: "fill--gray",
                alt: B.a
            });
            var Pa = r(627),
                Sa = r(479);

            function wa(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class _a extends l.PureComponent {
                constructor() {
                    super(...arguments), wa(this, "onChangeFilter", e => {
                        const t = "string" == typeof e ? e : e.target.value;
                        this.props.onChangeFilter && this.props.onChangeFilter(t);
                        const r = this.context.analyticsLocation;
                        if (r) {
                            const e = K.a.getIn(r, "FILTER_BY");
                            Object(ct.a)(Pa.a.create(e, {
                                filter: t,
                                heapId: Oe.f
                            }))
                        } else Object(ct.a)(Pa.a.create(K.a.FILTER_BY, {
                            filter: t,
                            heapId: Oe.f
                        }))
                    })
                }
                renderOptions() {
                    return c.a.createElement("select", {
                        className: "legacy__select",
                        onChange: this.onChangeFilter,
                        value: this.props.intent,
                        "data-test-id": "intent-filter-select"
                    }, this.props.options.map(e => c.a.createElement("option", {
                        value: e.id,
                        key: e.id
                    }, Object(F.c)(t => t.filter[e.id].label))))
                }
                render() {
                    const {
                        className: e,
                        labelText: t
                    } = this.props, r = w()(e, "search-control-button select--no-styles--no-flex"), a = this.context.envInfo.isMobile ? t : c.a.createElement(Sa.a, {
                        className: "flex-row-vcenter",
                        tooltipPosition: "bottom-left",
                        tooltipContent: c.a.createElement(fe.a, {
                            content: e => e.citations.intents.tooltip
                        })
                    }, t, c.a.createElement(va, {
                        className: "fill--gray",
                        height: "12",
                        width: "12"
                    }));
                    return c.a.createElement("div", {
                        className: r
                    }, c.a.createElement("label", {
                        className: "search-control-button-label"
                    }, a), this.renderOptions())
                }
            }
            wa(_a, "propTypes", {
                className: O.a.string,
                intent: O.a.string,
                labelText: O.a.string,
                onChangeFilter: O.a.func,
                options: O.a.instanceOf(Et.b.List).isRequired
            }), wa(_a, "contextTypes", {
                analyticsLocation: O.a.object,
                envInfo: O.a.instanceOf(b.a).isRequired
            }), wa(_a, "defaultProps", {
                labelText: "Citation Type:"
            });
            var Ca = r(494),
                xa = r(634),
                ja = r(635),
                Ta = r(442);

            function ka() {
                return (ka = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function Ia(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Na extends l.PureComponent {
                constructor() {
                    super(...arguments), Ia(this, "onChangeSort", e => {
                        const t = "string" == typeof e ? e : e.target.value;
                        this.props.onChangeSort && this.props.onChangeSort(t);
                        const r = this.context.analyticsLocation;
                        if (r) {
                            const e = K.a.getIn(r, "SORT_BY");
                            Object(ct.a)(Ca.a.create(e, {
                                sort: t
                            }))
                        } else Object(ct.a)(Ca.a.create(K.a.SORT_BY, {
                            sort: t
                        }))
                    })
                }
                renderOptions() {
                    return l.createElement("select", ka({
                        className: "legacy__select search-sort-select",
                        onChange: this.onChangeSort,
                        value: this.props.sort,
                        "data-test-id": "sort-select"
                    }, Object(Ta.d)()), this.props.options.map(e => l.createElement("option", {
                        value: e.id,
                        key: e.id
                    }, Object(F.c)(t => t.sort[e.id].label))))
                }
                renderTabs() {
                    const e = this.props.options.map(e => l.createElement(xa.a, {
                        key: e.id,
                        isActive: e.id === this.props.sort
                    }, l.createElement(Mt.a, {
                        iconProps: {
                            icon: e.icon,
                            width: "15",
                            height: "15"
                        },
                        text: Object(F.c)(t => t.sort[e.id].label),
                        onClick: this.onChangeSort.bind(this, e.id),
                        style: {
                            border: "none"
                        },
                        testId: "sort-by-" + e.id
                    })));
                    return l.createElement(ja.a, null, e.toArray())
                }
                render() {
                    const {
                        className: e,
                        displayType: t,
                        labelText: r
                    } = this.props, a = w()(e, "search-sort flex-row-vcenter author-publications-list-sorts");
                    return l.createElement("div", {
                        className: a
                    }, l.createElement("label", {
                        className: "search-sort-label"
                    }, r), "tabs" === t ? this.renderTabs() : this.renderOptions())
                }
            }

            function Da() {
                return (Da = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function Ra(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            Ia(Na, "contextTypes", {
                analyticsLocation: O.a.object
            }), Ia(Na, "defaultProps", {
                displayType: "select",
                labelText: "Sort by:"
            });
            class La extends c.a.PureComponent {
                constructor() {
                    super(...arguments), Ra(this, "onChangeSort", e => {
                        const t = "string" == typeof e ? e : e.target.value;
                        this.props.onChangeSort && this.props.onChangeSort(t);
                        const r = this.context.analyticsLocation;
                        if (r) {
                            const e = K.a.getIn(r, "SORT_BY");
                            Object(ct.a)(Ca.a.create(e, {
                                sort: t
                            }))
                        } else Object(ct.a)(Ca.a.create(K.a.SORT_BY, {
                            sort: t
                        }))
                    })
                }
                renderOptions() {
                    return c.a.createElement("select", Da({
                        className: "legacy__select",
                        onChange: this.onChangeSort,
                        value: this.props.sort,
                        "data-test-id": "sort-select"
                    }, Object(Ta.d)()), this.props.options.map(e => c.a.createElement("option", {
                        value: e.id,
                        key: e.id
                    }, Object(F.c)(t => t.sort[e.id].label))))
                }
                render() {
                    const {
                        className: e,
                        labelText: t
                    } = this.props, r = w()(e, "search-control-button select--no-styles--no-flex author-publications-list-sorts");
                    return c.a.createElement("div", {
                        className: r
                    }, c.a.createElement("label", {
                        className: "search-control-button-label"
                    }, t), this.renderOptions())
                }
            }
            Ra(La, "contextTypes", {
                analyticsLocation: O.a.object
            }), Ra(La, "defaultProps", {
                labelText: "Sort by:"
            });
            var Aa = r(18);

            function Fa(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Ma extends c.a.PureComponent {
                constructor() {
                    super(...arguments), Fa(this, "getQueryParametersForType", e => {
                        if ("citingPapers" === e) {
                            const e = this.context.citationQueryStore.getQuery();
                            return Object(Aa.a)(e)
                        }
                        const t = this.context.paperStore.getPaperDetail().get(e),
                            r = L.b.CITATIONS_PAGE_SIZE,
                            a = {};
                        a[e + "Sort"] = t.get("sort"), a[e + "Limit"] = r, a[e + "Offset"] = (t.pageNumber - 1) * r;
                        const n = t.get("citationIntent");
                        return "citingPapers" === e && n && n !== L.b.INTENTS.ALL_INTENTS.id && (a.citationIntent = n), "citingPapers" === e && t.yearFilter && (a.year = [t.yearFilter.get("min"), t.yearFilter.get("max")]), a
                    }), Fa(this, "getQueryParametersForAll", () => ot()(this.getQueryParametersForType("citingPapers"), this.getQueryParametersForType("citedPapers"))), Fa(this, "updateQuery", e => {
                        const t = {
                                paperId: this.context.paperStore.getPaperDetail().paper.id,
                                slug: this.context.paperStore.getPaperDetail().paper.slug
                            },
                            {
                                history: r
                            } = this.context;
                        r.push(Object(Ee.f)({
                            routeName: "PAPER_DETAIL",
                            params: t,
                            query: e
                        }))
                    }), Fa(this, "onChangeSort", e => {
                        const t = this.getQueryParametersForAll();
                        t[this.props.citationType + "Sort"] = e, this.updateQuery(t)
                    }), Fa(this, "onChangeIntentFilter", e => {
                        const t = this.getQueryParametersForAll();
                        e && e !== L.b.INTENTS.ALL_INTENTS.id ? t.citationIntent = e : delete t.citationIntent, this.updateQuery(t)
                    }), Fa(this, "renderListLabel", () => {
                        const {
                            citationPage: {
                                citationType: e,
                                loading: t,
                                pageNumber: r,
                                totalCitations: a,
                                citationIntent: n,
                                yearFilter: i
                            },
                            citationCoverage: o,
                            showSummary: l
                        } = this.props, p = i && i.get("min") && 0 !== i.get("min") || i && i.get("max") && 0 !== i.get("max") || n && "all" !== n && "citingPapers" === e;
                        if (!l) return null;
                        const u = p ? ne.j(a, Object(F.c)(t => t.citations.list.label.filtered[e]), Object(F.c)(t => t.citations.list.label.filtered[e + "Plural"])) : ne.j(a, Object(F.c)(t => t.citations.list.label[e]), Object(F.c)(t => t.citations.list.label[e + "Plural"])),
                            d = (r - 1) * L.b.CITATIONS_PAGE_SIZE + 1,
                            h = Math.min(r * L.b.CITATIONS_PAGE_SIZE, a),
                            m = Object(F.c)(e => e.citations.list.range, d, h, u),
                            b = o && !p ? Object(F.c)(e => e.citations.list.coverageEstimate, m, o) : m,
                            f = !o || p || this.context.envInfo.isMobile ? c.a.createElement("p", {
                                "aria-live": "polite",
                                "aria-atomic": "true"
                            }, b) : c.a.createElement(Sa.a, {
                                className: "flex-row-vcenter",
                                tooltipContent: Object(F.c)(e => e.citations.list.tooltip, o)
                            }, b, c.a.createElement(va, {
                                className: "fill--gray",
                                height: "12",
                                width: "12"
                            }));
                        return t ? c.a.createElement("div", {
                            className: "citation-list__label"
                        }, c.a.createElement("span", null, c.a.createElement(s.a, null), " Loading", " ")) : c.a.createElement("div", {
                            className: "citation-list__label",
                            "data-test-id": "citation-list-header"
                        }, f)
                    })
                }
                render() {
                    const {
                        allowRelevanceSort: e,
                        showSummary: t,
                        citationPage: {
                            citationIntent: r
                        }
                    } = this.props, a = this.context.envInfo.isMobile, n = "citingPapers" === this.props.citationType, i = r && "all" !== r, s = a ? "controls-split" : "controls-right", o = a ? "controls-full" : "controls-right", l = a || n ? null : "select--no-styles--inline", p = e ? se.a.citationsWithRelevance() : se.a.citations(), u = this.props.totalPages > 1 ? c.a.createElement("div", {
                        className: n ? s : o
                    }, c.a.createElement(La, {
                        onChangeSort: this.onChangeSort,
                        options: p,
                        sort: this.props.sort,
                        labelText: Object(F.c)(e => e.citations.list.sortByLabel),
                        className: l
                    })) : null, d = n && (this.props.totalPages > 1 || i) ? c.a.createElement("div", {
                        className: a ? "controls-split" : "controls-left"
                    }, c.a.createElement(_a, {
                        onChangeFilter: this.onChangeIntentFilter,
                        options: L.b.getIntents(),
                        intent: r,
                        labelText: Object(F.c)(e => e.citations.list.filterByIntentLabel)
                    })) : null;
                    if (n && !a) {
                        const e = t ? "controls" : "controls__no-line";
                        return c.a.createElement("div", null, c.a.createElement("div", {
                            className: e
                        }, d, u), t && this.props.citationPage.totalCitations > 1 ? this.renderListLabel() : null)
                    }
                    if (a) return t ? c.a.createElement("div", null, this.props.citationPage.totalCitations > 1 ? this.renderListLabel() : null) : c.a.createElement("div", null, c.a.createElement("div", {
                        className: "controls__no-line"
                    }, d, u));
                    if (!n && !a) {
                        const e = t ? "controls" : "controls__no-line";
                        return c.a.createElement("div", {
                            className: e
                        }, t && this.props.citationPage.totalCitations > 1 ? this.renderListLabel() : null, u)
                    }
                    return c.a.createElement("div", {
                        className: "controls"
                    }, this.props.citationPage.totalCitations > 1 && this.renderListLabel(), this.props.totalPages > 1 && c.a.createElement("div", {
                        className: a ? "" : "controls-right"
                    }, c.a.createElement(Na, {
                        onChangeSort: this.onChangeSort,
                        options: p,
                        sort: this.props.sort,
                        labelText: Object(F.c)(e => e.citations.list.sortByLabel)
                    })))
                }
            }
            Fa(Ma, "contextTypes", {
                citationQueryStore: O.a.instanceOf(I.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                paperStore: O.a.object.isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var qa = r(143);

            function Ba(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Va(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            const Ha = Object(We.taggedSoftError)("simpleCitationsList");
            class Qa extends c.a.PureComponent {
                constructor() {
                    super(...arguments), Va(this, "navigateToCitationPage", e => {
                        const {
                            citationQueryStore: t,
                            paperStore: r,
                            history: a
                        } = this.context, n = this.props.citationPage.citationType;
                        Object(qa.a)({
                            citationType: n,
                            pageNumber: e,
                            citationQueryStore: t,
                            paperStore: r,
                            history: a
                        }), this.props.shouldScrollOnPaginate && this.scrollOnPaginate(this.props.cardId)
                    }), Va(this, "scrollOnPaginate", e => {
                        const t = this.context.paperNavStore.getNavTarget(e);
                        t ? He.a.smoothScrollTo(t, () => {}) : Ha("Failed to scroll on paginate, getNavTarget did not find target for " + e)
                    }), Va(this, "trackClickCitationLink", e => {
                        const {
                            paper: t,
                            citationPage: r
                        } = this.props, {
                            citations: a,
                            citationType: n,
                            pageNumber: i,
                            sort: s
                        } = r;
                        Ye({
                            paperId: t.id,
                            section: "citingPapers" === n ? "inboundCitations" : "outboundCitations",
                            sortOrder: s,
                            page: i,
                            paperList: a,
                            clickedPaper: e,
                            swapPosition: void 0
                        })
                    }), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? Ba(Object(r), !0).forEach((function(t) {
                                Va(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Ba(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromWeblabStore()), this.context.weblabStore.registerComponent(this, () => {
                        this.setState(this.getStateFromWeblabStore())
                    })
                }
                getStateFromWeblabStore() {
                    const {
                        weblabStore: e
                    } = this.context;
                    return {
                        isPaperRowV2FontOnly: e.isVariationEnabled(Qe.b.PaperRowV2FontOnly.KEY, Qe.b.PaperRowV2FontOnly.Variation.PAPER_ROW_V2_FONT_ONLY)
                    }
                }
                renderCitationsListWithIntents() {
                    const {
                        paper: e,
                        citationPage: t
                    } = this.props, {
                        citations: r,
                        citationType: a
                    } = t, {
                        isPaperRowV2FontOnly: n
                    } = this.state;
                    return r.isEmpty() ? null : c.a.createElement("div", {
                        className: "citation-list__citations"
                    }, r.map((t, r) => {
                        const i = Object(_e.b)(t);
                        return c.a.createElement(Ie.c, {
                            key: [t.id, r].join("|"),
                            paper: i,
                            eventData: {
                                parentPaper: e,
                                index: r
                            },
                            onClickTitle: !1
                        }, c.a.createElement(Be.a, null, c.a.createElement(Fe.default, {
                            paper: i,
                            className: w()("citation-list__paper-row", {
                                "paper-v2-font-only": n
                            }),
                            title: c.a.createElement(Me.default, {
                                paper: i,
                                onClick: () => this.trackClickCitationLink(t),
                                testId: "citation-paper-title",
                                heapProps: {
                                    id: Oe.g,
                                    "paper-id": t.id,
                                    "citation-type": a,
                                    "has-intents": t.citationContexts.size > 0
                                }
                            }),
                            meta: c.a.createElement(Ae.default, {
                                paper: i,
                                authors: !i.authors.isEmpty() && c.a.createElement(Ne.default, {
                                    paper: i,
                                    heapProps: {
                                        id: Oe.d
                                    }
                                })
                            }),
                            controls: c.a.createElement(Le.default, {
                                paper: i,
                                stats: t.numCitedBy > 0 && c.a.createElement(xe, {
                                    citation: t,
                                    citationType: a,
                                    citedPaperTitle: e.title
                                }),
                                flags: c.a.createElement(ge.default, {
                                    citation: t,
                                    citationType: a,
                                    citedPaperTitle: e.title,
                                    shouldRenderIntents: !0,
                                    className: "cl-paper-controls__flags"
                                })
                            }),
                            abstract: c.a.createElement(Ge.b, {
                                paper: i
                            })
                        })), c.a.createElement(Ve.a, null, c.a.createElement(De.default, {
                            paper: i,
                            title: c.a.createElement(Me.default, {
                                paper: i,
                                onClick: () => this.trackClickCitationLink(t),
                                testId: "citation-paper-title",
                                heapProps: {
                                    id: Oe.g,
                                    "paper-id": t.id,
                                    "citation-type": a,
                                    "has-intents": t.citationContexts.size > 0
                                }
                            }),
                            meta: c.a.createElement(Ae.default, {
                                paper: i,
                                shouldStackMeta: !0,
                                authors: !i.authors.isEmpty() && c.a.createElement(Ne.default, {
                                    paper: i,
                                    heapProps: {
                                        id: Oe.d
                                    }
                                })
                            }),
                            controls: t.numCitedBy > 0 && c.a.createElement(Le.default, {
                                paper: i,
                                actions: !1,
                                stats: c.a.createElement(xe, {
                                    citation: t,
                                    citationType: a,
                                    citedPaperTitle: e.title
                                })
                            }),
                            className: w()("citation-list__paper-card", {
                                "paper-v2-font-only": n
                            }),
                            header: c.a.createElement(ge.default, {
                                citation: t,
                                citationType: a,
                                citedPaperTitle: e.title,
                                shouldRenderIntents: !0
                            }),
                            footer: c.a.createElement(Re.a, {
                                paper: i
                            }),
                            abstract: c.a.createElement(Ge.b, {
                                paper: i,
                                className: "tldr__paper-card"
                            })
                        })))
                    }))
                }
                renderEmptyMessage() {
                    const {
                        citationType: e,
                        citationIntent: t,
                        yearFilter: r
                    } = this.props.citationPage, a = t && "all" !== t || r && r.get("min") && 0 !== r.get("min") || r && r.get("max") && 0 !== r.get("max");
                    return c.a.createElement(Ea, {
                        type: e,
                        isFiltered: a
                    })
                }
                render() {
                    const e = this.context.envInfo.isMobile,
                        {
                            citationPage: t
                        } = this.props,
                        r = t.loading,
                        a = t.totalPages > 1 ? c.a.createElement("div", {
                            className: "citation-pagination flex-row-vcenter"
                        }, r ? c.a.createElement("span", {
                            className: "flex-row-vcenter citation-loading"
                        }, c.a.createElement(s.a, null), " Loading") : null, c.a.createElement(ke.default, {
                            size: e ? ke.SIZE.LARGE : ke.SIZE.DEFAULT,
                            maxVisiblePageButtons: e ? 4 : 5,
                            pageNumber: this.props.citationPage.pageNumber,
                            onPaginate: this.navigateToCitationPage,
                            totalPages: Math.min(L.b.MAX_CITATION_PAGES, t.totalPages)
                        })) : null,
                        n = this.props.citationPage.totalCitations > 0 ? this.renderCitationsListWithIntents() : this.renderEmptyMessage(),
                        i = c.a.createElement(Ma, {
                            allowRelevanceSort: this.props.allowRelevanceSort,
                            paper: this.props.paper,
                            citationCoverage: this.props.citationCoverage,
                            citationPage: this.props.citationPage,
                            sort: t.sort,
                            totalPages: t.totalPages,
                            citationType: t.citationType,
                            useButtons: !e,
                            showSummary: !0
                        });
                    return c.a.createElement("div", {
                        className: "paper-detail-content-card",
                        "data-test-id": "citingPapers" === t.citationType ? "cited-by" : "reference"
                    }, i, n, a)
                }
            }
            Va(Qa, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired,
                citationQueryStore: O.a.instanceOf(I.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                paperNavStore: O.a.instanceOf(Ue.a).isRequired,
                paperStore: O.a.instanceOf(qa.b).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var Ua = Object(Ke.b)(K.a.PaperDetail.References)(Qa),
                za = r(598);

            function Ya() {
                return (Ya = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function Wa(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Ga extends l.PureComponent {
                constructor() {
                    super(...arguments), Wa(this, "state", {
                        isToggleOpen: !1
                    }), Wa(this, "toggleIcon", () => {
                        const e = !this.state.isToggleOpen;
                        this.setState({
                            isToggleOpen: e
                        })
                    })
                }
                render() {
                    const {
                        children: e,
                        icon: t,
                        isMobile: r,
                        mobileToggle: a,
                        title: n
                    } = this.props;
                    let i;
                    if (r && a) {
                        const e = this.state.isToggleOpen;
                        i = {
                            icon: e ? "x" : t,
                            className: e ? "paper-card__stats-item toggle-open" : "paper-card__stats-item",
                            isCollapsible: !0,
                            startsOpen: !0,
                            onToggle: this.toggleIcon
                        }
                    } else i = {
                        className: "paper-card__stats-item"
                    };
                    return l.createElement(za.a, Ya({
                        title: n ? n.toUpperCase() : null
                    }, i), e)
                }
            }

            function Ka(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class $a extends l.PureComponent {
                static getTabLabel(e) {
                    var t;
                    const r = (null != (t = e) && null != (t = t.paperDetail) && null != (t = t.citedPapers) ? t.totalCitations : t) || 0;
                    return Object(F.a)(e => e.paperDetail.tabLabels.referencedPapers, r)
                }
                static hasContent(e) {
                    var t;
                    return ((null != (t = e) && null != (t = t.paperDetail) && null != (t = t.citedPapers) ? t.totalCitations : t) || 0) > 0
                }
                render() {
                    var e;
                    const {
                        isMobile: t
                    } = this.context.envInfo, {
                        paper: r,
                        paper: {
                            citationStats: {
                                numViewableReferences: a
                            }
                        },
                        citedPapers: n,
                        citationSortAvailability: i
                    } = this.props.paperDetail, s = this.constructor.getTabLabel(this.props), o = (null != (e = i) ? e.citedPapers : e) && i.citedPapers.includes(se.a.RELEVANCE.id);
                    return t ? l.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: Object(F.c)(e => e.paperDetail.sectionTitles.referencedPapers)
                    }, l.createElement(_, {
                        cardId: $a.CARD_ID,
                        navLabel: s,
                        className: "references"
                    }, l.createElement(k, {
                        title: Object(F.c)(e => e.paperDetail.sectionTitles.referencedPapers)
                    }), l.createElement(x, {
                        willChildrenOwnLayout: !0
                    }, l.createElement(wt, null, l.createElement(Ua, {
                        allowRelevanceSort: o,
                        shouldScrollOnPaginate: !0,
                        cardId: $a.CARD_ID,
                        name: "citedPapers",
                        paper: r,
                        citationPage: n,
                        citationTotal: a
                    })), l.createElement(Oa, null, l.createElement("div", {
                        className: "card-callout__flush-top card-callout__mobile"
                    }, l.createElement("div", {
                        className: "paper-detail__card__filter-block"
                    }, l.createElement(Ga, null, l.createElement(Ma, {
                        allowRelevanceSort: o,
                        cardId: $a.CARD_ID,
                        shouldScrollOnPaginate: !0,
                        paper: r,
                        citationPage: n,
                        sort: n.sort,
                        totalPages: n.totalPages,
                        citationType: n.citationType,
                        useButtons: !0,
                        showSummary: !1
                    })))))))) : l.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: Object(F.c)(e => e.paperDetail.sectionTitles.referencedPapers)
                    }, l.createElement(_, {
                        cardId: $a.CARD_ID,
                        navLabel: s,
                        className: "references"
                    }, l.createElement(k, {
                        title: Object(F.c)(e => e.paperDetail.sectionTitles.referencedPapers)
                    }), l.createElement(x, null, l.createElement(Ua, {
                        allowRelevanceSort: o,
                        shouldScrollOnPaginate: !0,
                        cardId: $a.CARD_ID,
                        name: "citedPapers",
                        paper: r,
                        citationPage: n,
                        citationTotal: a
                    }))))
                }
            }
            Ka($a, "CARD_ID", "references"), Ka($a, "contextTypes", {
                envInfo: O.a.instanceOf(b.a).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var Xa = r(586),
                Za = r(512);
            class Ja extends l.PureComponent {
                render() {
                    return this.props.children
                }
            }
            var en = r(26),
                tn = r(192);

            function rn(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function an(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class nn extends l.PureComponent {
                componentDidMount() {
                    const e = function(e) {
                            for (var t = 1; t < arguments.length; t++) {
                                var r = null != arguments[t] ? arguments[t] : {};
                                t % 2 ? rn(Object(r), !0).forEach((function(t) {
                                    an(e, t, r[t])
                                })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : rn(Object(r)).forEach((function(t) {
                                    Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                                }))
                            }
                            return e
                        }({}, this.context.heapPropsChain || {}, {}, {
                            experiment: this.props.experiment.KEY,
                            impressedAs: this.props.impressedAs
                        }),
                        t = pr.a.create("experiment-impression", e);
                    Object(tn.a)([t])
                }
                render() {
                    return l.createElement(i.a, {
                        heapProps: {
                            "research-experiment": this.props.experiment.KEY,
                            "research-experiment-impressed-as": this.props.impressedAs
                        }
                    }, this.props.children)
                }
            }
            an(nn, "contextTypes", {
                heapPropsChain: O.a.object
            });
            var sn = r(604),
                on = r.n(sn);

            function ln(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function cn(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            const pn = function(e) {
                for (var t = 1; t < arguments.length; t++) {
                    var r = null != arguments[t] ? arguments[t] : {};
                    t % 2 ? ln(Object(r), !0).forEach((function(t) {
                        cn(e, t, r[t])
                    })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ln(Object(r)).forEach((function(t) {
                        Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                    }))
                }
                return e
            }({}, Ce.b.ChildContextTypes, {
                heapProps: O.a.object,
                heapPropsChain: O.a.object,
                private: O.a.object,
                router: O.a.object
            });
            class un extends c.a.PureComponent {
                getChildContext() {
                    return this.props.context
                }
                render() {
                    return this.props.children
                }
            }
            cn(un, "childContextTypes", pn);
            class dn extends c.a.PureComponent {
                constructor() {
                    super(...arguments), this.state = {
                        elementToRender: this.props.children
                    }
                }
                componentWillReceiveProps(e) {
                    this.setState({
                        elementToRender: e.children
                    })
                }
                componentDidCatch(e, t) {
                    this.props.onCatch && this.props.onCatch(e, t), this.setState({
                        elementToRender: this.props.fallback
                    })
                }
                render() {
                    if (Object(jt.a)()) return c.a.createElement("div", {
                        key: "ssr-boundary"
                    }, this.state.elementToRender);
                    try {
                        const e = on.a.renderToStaticMarkup(c.a.createElement(un, {
                            context: this.context
                        }, c.a.createElement(Ce.a.Provider, {
                            value: this.context
                        }, this.props.children)));
                        return c.a.createElement("div", {
                            key: "ssr-boundary",
                            dangerouslySetInnerHTML: {
                                __html: e
                            }
                        })
                    } catch (e) {
                        return Object(We.default)("serversideerrorboundary.fallback", `Error during render, falling back. Cause: [message=${e.message}]`, e), c.a.createElement("div", {
                            key: "ssr-boundary"
                        }, this.props.fallback)
                    }
                }
            }
            cn(dn, "contextTypes", pn);
            var hn = r(513);

            function mn(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function bn(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class fn extends l.PureComponent {
                constructor() {
                    super(...arguments), bn(this, "onCatch", (e, t) => {
                        const r = function(e) {
                            for (var t = 1; t < arguments.length; t++) {
                                var r = null != arguments[t] ? arguments[t] : {};
                                t % 2 ? mn(Object(r), !0).forEach((function(t) {
                                    bn(e, t, r[t])
                                })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : mn(Object(r)).forEach((function(t) {
                                    Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                                }))
                            }
                            return e
                        }({}, t, {
                            experimentKey: this.props.experiment.KEY,
                            variationName: this.state.isEligible ? this.context.weblabStore.getVariation(this.props.experiment.KEY) : "fallback"
                        });
                        Object(en.a)().logNamedError("error.skipper.render", e, r)
                    });
                    const e = !this.props.eligibilityTest || this.props.eligibilityTest();
                    e && this.context.weblabStore.expose(this.props.experiment.KEY), this.state = {
                        isEligible: e
                    }
                }
                renderVariationWithImpression(e) {
                    const {
                        children: t,
                        variation: r
                    } = e.props;
                    return l.createElement(hn.a, {
                        variation: r,
                        key: r
                    }, l.createElement(nn, {
                        experiment: this.props.experiment,
                        impressedAs: r
                    }, gn(t)))
                }
                renderFallbackWithImpression(e) {
                    var t;
                    return l.createElement(Ja, null, l.createElement(nn, {
                        experiment: this.props.experiment,
                        impressedAs: "fallback"
                    }, gn(null != (t = e) && null != (t = t.props) ? t.children : t)))
                }
                renderIneligible() {
                    var e;
                    const t = l.Children.toArray(this.props.children).find(e => e.type === Za.a) || null;
                    return l.createElement(nn, {
                        experiment: this.props.experiment,
                        impressedAs: "ineligible"
                    }, gn(null != (e = t) && null != (e = e.props) ? e.children : e))
                }
                getFallbackAndVariations() {
                    const e = l.Children.toArray(this.props.children);
                    let t = null;
                    const r = [];
                    return e.forEach(e => {
                        e.type === Ja ? t = e : (r.push(e), t || e.type !== Za.a || (t = e))
                    }), {
                        fallback: t,
                        variations: r
                    }
                }
                renderExperiment() {
                    const {
                        fallback: e,
                        variations: t
                    } = this.getFallbackAndVariations(), r = this.renderFallbackWithImpression(e);
                    return l.createElement(dn, {
                        fallback: r,
                        onCatch: this.onCatch
                    }, l.createElement(Xa.a, {
                        experiment: this.props.experiment
                    }, t.map(e => this.renderVariationWithImpression(e))))
                }
                render() {
                    return this.state.isEligible ? this.renderExperiment() : this.renderIneligible()
                }
            }

            function gn(e) {
                return e ? "function" == typeof e ? e() : e : null
            }
            bn(fn, "contextTypes", {
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var yn = r(374),
                On = r(625),
                En = r(63);

            function vn(e) {
                let {
                    paperDetail: t
                } = e;
                const r = t.paper.authors.map(e => e.alias),
                    {
                        envInfo: a
                    } = Object(Ce.d)();
                return l.createElement("div", {
                    className: w()({
                        "pdp-author-claim": !0,
                        "pdp-author-claim--is-mobile": a.isMobile
                    })
                }, l.createElement("div", {
                    className: "pdp-author-claim__headline"
                }, Object(F.c)(e => e.paperDetail.authorClaim.headline)), l.createElement("ul", {
                    className: "pdp-author-claim__author-list"
                }, r.map(e => {
                    const t = Object(En.c)(e);
                    return t ? l.createElement("li", {
                        id: t,
                        className: "pdp-author-claim__author-list-item"
                    }, e.isClaimed() ? l.createElement(yn.default, {
                        className: "pdp-author-claim__claimed-author",
                        icon: l.createElement(On.a, {
                            sizePx: 12
                        }),
                        label: e.name
                    }) : l.createElement(Tt.a, {
                        to: "AUTHOR_CLAIM",
                        params: {
                            authorId: t,
                            slug: e.slug
                        },
                        className: "pdp-author-claim__claim-link"
                    }, l.createElement(N.default, {
                        className: "pdp-author-claim__author-button",
                        label: e.name
                    }))) : null
                })))
            }

            function Pn(e) {
                const {
                    fieldsOfStudy: t
                } = e.paper;
                if (!t || !t.size) return null;
                const r = t.join(", ");
                return l.createElement(l.Fragment, null, r)
            }

            function Sn(e) {
                let {
                    childList: t,
                    heapId: r
                } = e;
                return l.createElement("ul", {
                    className: "flex-row-vcenter paper-meta",
                    "data-heap-id": r
                }, t && t.map((e, t) => e && l.createElement("li", {
                    className: "paper-meta-item",
                    key: t
                }, e)))
            }
            var wn = r(521);

            function _n(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Cn(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class xn extends l.PureComponent {
                constructor() {
                    super(...arguments), Cn(this, "onClickDeepLink", () => {
                        const {
                            citationQueryStore: e,
                            router: t,
                            history: r
                        } = this.context, {
                            citationIntent: a
                        } = this.props, {
                            query: n
                        } = this.state, i = a === L.b.ALL ? se.a.IS_INFLUENTIAL_CITATION.id : se.a.RELEVANCE.id;
                        ie.a.changeRouteForPartialQuery(e.getIndependentQuery(), n.set("citationIntent", a).set("sort", i), r, t)
                    }), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? _n(Object(r), !0).forEach((function(t) {
                                Cn(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : _n(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromCitationQueryStore()), this.context.citationQueryStore.registerComponent(this, () => {
                        this.setState(this.getStateFromCitationQueryStore())
                    })
                }
                getStateFromCitationQueryStore() {
                    return {
                        query: this.context.citationQueryStore.getQuery(),
                        queryText: this.context.citationQueryStore.getQuery().queryString || "",
                        queryResponse: this.context.citationQueryStore.getQueryResponse(),
                        statsResponse: this.context.citationQueryStore.getAggregationResponse(),
                        isLoading: this.context.citationQueryStore.isLoading(),
                        isFiltering: this.context.citationQueryStore.isFiltering(),
                        isAggsLoading: this.context.citationQueryStore.isAggsLoading()
                    }
                }
                render() {
                    const {
                        title: e,
                        intentCount: t,
                        isKeyCitationType: r,
                        navId: a
                    } = this.props, n = r ? "scorecard-citation__key-title" : "scorecard-citation__title";
                    return l.createElement("div", {
                        className: "scorecard-citation__metadata"
                    }, l.createElement("div", {
                        className: "scorecard-citation__metadata__stat"
                    }, l.createElement(ma.a, {
                        navId: a,
                        onClick: this.onClickDeepLink
                    }, l.createElement("div", {
                        className: n
                    }, e)), l.createElement("span", null, r && l.createElement(Tt.a, {
                        "aria-label": Object(F.c)(e => e.paperDetail.scorecard.highlyInfluential.faqAriaLabel),
                        to: "FAQ_ROOT",
                        hash: "influential-citations"
                    }, l.createElement(ue.a, {
                        icon: "information",
                        className: "scorecard-citation__influential-citations-info"
                    })))), l.createElement("div", {
                        className: "scorecard-citation__metadata-item"
                    }, t.toLocaleString()))
                }
            }

            function jn(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Tn(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            Cn(xn, "contextTypes", {
                citationQueryStore: O.a.instanceOf(I.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                router: O.a.object.isRequired
            });
            class kn extends l.PureComponent {
                constructor() {
                    super(...arguments), Tn(this, "onClick", () => {
                        const {
                            onClick: e
                        } = this.props;
                        e && e()
                    })
                }
                render() {
                    const {
                        headline: e,
                        description: t,
                        tooltip: r,
                        background: a,
                        type: n
                    } = this.props, i = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? jn(Object(r), !0).forEach((function(t) {
                                Tn(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : jn(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({
                        className: w()({
                            scorecard__stat: !0,
                            ["scorecard__" + n]: !0
                        })
                    }, Object(ye.a)({
                        id: Oe.n,
                        type: n
                    }), {
                        "data-test-id": "scorecard-" + n,
                        onClick: this.onClick
                    });
                    return l.createElement("div", i, a && l.createElement("div", {
                        className: "scorecard__background"
                    }, a), l.createElement("div", {
                        className: "scorecard-stat__body",
                        "data-test-id": "scorecard-body"
                    }, e && l.createElement("span", {
                        className: "scorecard-stat__headline"
                    }, e), r && l.createElement("div", {
                        className: "scorecard-stat__tooltip"
                    }, r), t && l.createElement("div", {
                        className: "scorecard__description"
                    }, t)))
                }
            }

            function In() {
                return (In = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }
            class Nn extends l.PureComponent {
                constructor() {
                    super(...arguments), this.state = {
                        navId: Ot.CARD_ID
                    }
                }
                componentDidMount() {
                    const {
                        stat: {
                            typeKey: e
                        }
                    } = this.props;
                    Object(ct.a)(pr.a.create(K.a.PaperDetail.SCORECARD_IMPRESSION, {
                        scorecardType: e
                    }))
                }
                renderHeadline() {
                    var e;
                    const t = (null != (e = this.props.paperDetail) && null != (e = e.paper) && null != (e = e.citationStats) ? e.numCitations : e) || 0;
                    return l.createElement(l.Fragment, null, l.createElement("span", {
                        className: "scorecard-stat__headline__dark"
                    }, Object(F.a)(e => e.paperDetail.scorecard.citedBy.headline, t)))
                }
                renderKeyCitation() {
                    const e = this.props.stat.keyCitationCount,
                        t = Object(F.c)(e => e.paperDetail.scorecard.highlyInfluential.title);
                    return l.createElement(xn, {
                        title: t,
                        intentCount: e,
                        isKeyCitationType: !0,
                        navId: this.state.navId,
                        citationIntent: L.b.ALL
                    })
                }
                renderCitationIntents() {
                    const e = this.props.stat.citationIntentCount,
                        t = e => e > 0;
                    return l.createElement("div", {
                        className: "scorecard-citation"
                    }, t(e.background) && l.createElement(xn, {
                        title: Object(F.c)(e => e.paperDetail.scorecard.citationIntent.background),
                        intentCount: e.background,
                        navId: this.state.navId,
                        citationIntent: L.b.BACKGROUND
                    }), t(e.methodology) && l.createElement(xn, {
                        title: Object(F.c)(e => e.paperDetail.scorecard.citationIntent.methodology),
                        intentCount: e.methodology,
                        navId: this.state.navId,
                        citationIntent: L.b.METHODOLOGY
                    }), t(e.result) && l.createElement(xn, {
                        title: Object(F.c)(e => e.paperDetail.scorecard.citationIntent.result),
                        intentCount: e.result,
                        navId: this.state.navId,
                        citationIntent: L.b.RESULT
                    }))
                }
                renderDescription() {
                    const e = this.props.stat.keyCitationCount > 0;
                    return l.createElement(l.Fragment, null, e && this.renderKeyCitation(), this.renderCitationIntents(), this.renderCitationPageButton())
                }
                renderCitationPageButton() {
                    return l.createElement(ma.a, {
                        className: "scorecard-stat__headline__view-all",
                        navId: "citing-papers"
                    }, l.createElement(N.default, In({
                        className: "scorecard-stat__headline__view-all_btn",
                        type: N.TYPE.SECONDARY,
                        size: N.SIZE.DEFAULT,
                        label: Object(F.c)(e => e.paperDetail.scorecard.highlyInfluential.viewAll)
                    }, Object(ta.b)({
                        key: "view_all"
                    }))))
                }
                render() {
                    return l.createElement(kn, {
                        type: "citation-intent",
                        headline: this.renderHeadline(),
                        description: this.renderDescription()
                    })
                }
            }

            function Dn() {
                return (Dn = Object.assign ? Object.assign.bind() : function(e) {
                    for (var t = 1; t < arguments.length; t++) {
                        var r = arguments[t];
                        for (var a in r) Object.prototype.hasOwnProperty.call(r, a) && (e[a] = r[a])
                    }
                    return e
                }).apply(this, arguments)
            }

            function Rn(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Ln(e) {
                for (var t = 1; t < arguments.length; t++) {
                    var r = null != arguments[t] ? arguments[t] : {};
                    t % 2 ? Rn(Object(r), !0).forEach((function(t) {
                        An(e, t, r[t])
                    })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : Rn(Object(r)).forEach((function(t) {
                        Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                    }))
                }
                return e
            }

            function An(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Fn extends l.Component {
                constructor() {
                    super(...arguments), this.state = Ln({}, this.getStateFromWeblabStore()), this.context.weblabStore.registerComponent(this, () => {
                        this.setState(Ln({}, this.getStateFromWeblabStore()))
                    })
                }
                getStateFromWeblabStore() {
                    return {
                        isHighlyInfluentialCitationsEnabled: this.context.weblabStore.isFeatureEnabled(at.b.HighlyInfluentialCitationsScorecard)
                    }
                }
                renderStat(e) {
                    const {
                        isHighlyInfluentialCitationsEnabled: t
                    } = this.state, {
                        numCitations: r
                    } = this.props.paperDetail.paper.citationStats;
                    return t && e.typeKey === A.a && r > 0 ? l.createElement(Nn, Dn({
                        stat: e
                    }, this.props)) : null
                }
                render() {
                    const {
                        paperDetail: {
                            paper: {
                                scorecardStats: e
                            }
                        }
                    } = this.props;
                    return !e || e.isEmpty() ? null : l.createElement("div", {
                        className: "scorecard"
                    }, l.createElement("div", {
                        className: "scorecard_container"
                    }, e.map((e, t) => {
                        const r = this.renderStat(e);
                        return null === r ? null : l.createElement(Dt.a, {
                            key: `${e.typeKey}-${t}`,
                            className: "scorecard__item"
                        }, r)
                    })))
                }
            }
            An(Fn, "contextTypes", {
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var Mn = r(534),
                qn = r(466),
                Bn = r(436),
                Vn = r(419),
                Hn = r(423),
                Qn = r(13),
                Un = r(518);

            function zn(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function Yn(e) {
                for (var t = 1; t < arguments.length; t++) {
                    var r = null != arguments[t] ? arguments[t] : {};
                    t % 2 ? zn(Object(r), !0).forEach((function(t) {
                        Wn(e, t, r[t])
                    })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : zn(Object(r)).forEach((function(t) {
                        Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                    }))
                }
                return e
            }

            function Wn(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Gn extends l.PureComponent {
                constructor() {
                    super(...arguments), Wn(this, "trackClickSimilarPapersLink", e => {
                        Ye({
                            paperId: this.props.paperDetail.paper.id,
                            section: "similarPapers",
                            paperList: this.props.paperDetail.relatedPapers,
                            swapPosition: null,
                            clickedPaper: e
                        });
                        const t = {
                            paperId: this.props.paperDetail.paper.id,
                            clickedPaperId: e.id
                        };
                        this.props.paperDetail.relatedPapers.forEach((function(e, r) {
                            t["paper" + r] = e.id
                        })), Object(tn.a)([new xt.a(K.a.PaperDetail.SIMILAR_PAPERS_LIST, t)])
                    }), Wn(this, "trackActionClick", (e, t) => {
                        const r = this.context.analyticsLocation,
                            a = xt.a;
                        if (r) {
                            const n = K.a.getIn(r, "Action", e),
                                i = Yn({}, t);
                            Object(ct.a)(a.create(n, i))
                        } else Tr.a.warn("Unable to track ClickEvent from component <SimilarPapersCardBase /> because it doesn't have an analytics location")
                    }), this.state = Yn({
                        papersHaveBeenRequested: !this.props.paperDetail.relatedPapers.isEmpty()
                    }, this.getStateFromStores(), {}, this.getStateFromWeblabStore()), this.context.paperNavStore.registerComponent(this, () => {
                        this.setState(Yn({}, this.state, {}, this.getStateFromStores()))
                    }), this.context.weblabStore.registerComponent(this, () => {
                        this.setState(this.getStateFromWeblabStore())
                    })
                }
                getStateFromWeblabStore() {
                    const {
                        weblabStore: e
                    } = this.context;
                    return {
                        isPaperRowV2FontOnly: e.isVariationEnabled(Qe.b.PaperRowV2FontOnly.KEY, Qe.b.PaperRowV2FontOnly.Variation.PAPER_ROW_V2_FONT_ONLY)
                    }
                }
                componentDidMount() {
                    this.state.isVisibleOrActive && this.loadRelatedPapers()
                }
                componentDidUpdate() {
                    this.state.isVisibleOrActive && this.loadRelatedPapers()
                }
                getStateFromStores() {
                    const e = this.context.paperNavStore,
                        t = e.navItems.get("related-papers");
                    return {
                        isVisibleOrActive: t && (e.isItemVisible(t) || e.isItemActive(t))
                    }
                }
                loadRelatedPapers() {
                    const {
                        paperDetail: {
                            paper: e
                        },
                        paperFetchCount: t,
                        paperRecommendationType: r
                    } = this.props, {
                        api: a
                    } = this.context;
                    e.id && !this.state.papersHaveBeenRequested && (setTimeout(() => {
                        a.fetchRelatedPapers(e.id, {
                            limit: t,
                            recommenderType: r
                        })
                    }, 0), this.setState({
                        papersHaveBeenRequested: !0
                    }))
                }
                renderMetaDataBottomFooter(e, t) {
                    return l.createElement("div", {
                        className: "similar-papers__footer"
                    }, l.createElement("span", {
                        className: "similar-papers__footer-left"
                    }, e.citationStats && e.citationStats.numCitations > 0 && l.createElement("span", {
                        className: "similar-papers__footer-text"
                    }, l.createElement("span", {
                        className: "similar-papers__footer-counts"
                    }, Object(ne.d)(e.citationStats.numCitations)), Object(ne.j)(e.citationStats.numCitations, Object(F.c)(e => e.paperDetail.relatedPapers.citation.singular), Object(F.c)(e => e.paperDetail.relatedPapers.citation.plural), !0)), e.citationStats && e.citationStats.numKeyCitations > 0 && l.createElement("span", {
                        className: "similar-papers__footer-text"
                    }, l.createElement("span", {
                        className: "similar-papers__footer-counts"
                    }, Object(ne.d)(e.citationStats.numKeyCitations)), Object(ne.j)(e.citationStats.numKeyCitations, Object(F.c)(e => e.paperDetail.relatedPapers.highlyInfluenced.singular), Object(F.c)(e => e.paperDetail.relatedPapers.highlyInfluenced.plural), !0))), l.createElement("div", {
                        className: "similar-papers__footer-right"
                    }, l.createElement(Rr, {
                        buttonText: Object(F.c)(e => e.paper.action.saveShort),
                        className: "similar-papers__metadata-save",
                        paper: e,
                        saveButtonIcon: "fa-bookmark-not-filled",
                        trackLibraryClick: () => this.trackActionClick("READING_LIST", {
                            action: "save",
                            paperId: e.id,
                            index: t
                        })
                    })))
                }
                renderLoading() {
                    const {
                        offSet: e
                    } = this.props;
                    return l.createElement(te.a, {
                        className: "similar-papers__body"
                    }, [...Array(e)].map((e, t) => l.createElement(Dt.a, {
                        className: "similar-papers__card",
                        key: t
                    }, l.createElement("div", {
                        className: "similar-papers-card__fold"
                    }), l.createElement("div", {
                        className: "similar-papers__shimmer"
                    }))))
                }
                getCurrentPaperSet(e) {
                    const {
                        offSet: t,
                        currentIndex: r,
                        smallScreenOffSet: a,
                        showMoreToggle: n,
                        isNarrowScreen: i
                    } = this.props, s = ei(e, t), o = s.size, l = r + t;
                    let c;
                    return c = r <= o - 1 && r > o - t ? s.slice(o - t, o) : r <= 0 ? s.slice(0, t) : s.slice(r, l), n && i ? c = s.slice(0, a) : i && !n && (c = s), c
                }
                getAbstractText(e) {
                    return e.tldr && e.tldr.text.length < 400 ? l.createElement(Un.a, {
                        paper: e
                    }) : e.paperAbstractTruncated ? l.createElement(Bn.default, {
                        abstract: Object(Qn.b)({
                            text: e.paperAbstractTruncated
                        })
                    }) : null
                }
                renderRelatedPaperFooter(e) {
                    return this.context.envInfo.isMobile ? l.createElement(Re.a, {
                        paper: e
                    }) : l.createElement("div", {
                        className: "similar-papers__paper-card-footer"
                    }, l.createElement(Hn.default, {
                        paper: e
                    }), l.createElement(Vn.default, {
                        paper: e,
                        viewPaper: !1,
                        cite: !1
                    }))
                }
                renderRelatedPapers(e) {
                    const t = this.getCurrentPaperSet(e),
                        r = this.context.envInfo.isMobile,
                        {
                            isPaperRowV2FontOnly: a
                        } = this.state;
                    return l.createElement(te.a, {
                        testId: "related-papers-list",
                        className: "similar-papers__body"
                    }, e.isEmpty() ? l.createElement("div", {
                        className: "empty"
                    }, Object(F.c)(e => e.paperDetail.relatedPapers.noPapers)) : t.map((e, t) => l.createElement(Ie.c, {
                        key: t,
                        paper: e,
                        eventData: {
                            index: t
                        },
                        onClickTitle: () => this.trackClickSimilarPapersLink(e)
                    }, l.createElement(De.default, {
                        paper: e,
                        className: w()("similar-papers__paper-card", {
                            "paper-v2-font-only": a
                        }),
                        hasDogEar: !0,
                        controls: !!r && l.createElement(Hn.default, {
                            paper: e
                        }),
                        abstract: this.getAbstractText(e),
                        footer: this.renderRelatedPaperFooter(e)
                    }))))
                }
                render() {
                    const {
                        paperDetail: {
                            isLoadingRelatedPapers: e,
                            relatedPapers: t
                        }
                    } = this.props;
                    return e ? this.renderLoading() : this.renderRelatedPapers(t)
                }
            }
            Wn(Gn, "contextTypes", {
                analyticsLocation: O.a.object,
                api: O.a.instanceOf(m.a).isRequired,
                paperNavStore: O.a.instanceOf(Ue.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            }), Wn(Gn, "defaultProps", {
                currentIndex: 0,
                paperFetchCount: 10,
                offSet: 3,
                isMobile: !1,
                mobileOffSet: 2,
                showMoreTruncated: !0
            });
            var Kn = Object(Ke.b)(K.a.PaperDetail.RelatedPapers)(Gn);
            class $n extends l.PureComponent {
                isActive() {
                    const {
                        indicatorIndex: e,
                        offSet: t,
                        paperCount: r,
                        currentIndex: a
                    } = this.props;
                    return function(e) {
                        let {
                            currentIndex: t,
                            indicatorIndex: r,
                            offSet: a,
                            paperCount: n
                        } = e;
                        return t <= 0 ? r >= 0 && r < a : r >= t && r < t + a || t >= n - a && t <= n - 1 && (r >= t || r >= n - a)
                    }({
                        indicatorIndex: e,
                        offSet: t,
                        paperCount: r,
                        currentIndex: a
                    })
                }
                render() {
                    return l.createElement("li", {
                        className: this.isActive() ? "carousel-indicator carousel-indicator-active" : "carousel-indicator"
                    })
                }
            }
            var Xn = r(658);

            function Zn(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class Jn extends c.a.PureComponent {
                constructor() {
                    super(...arguments), Zn(this, "_carouselContainerRef", void 0), Zn(this, "selectSizeRender", () => {
                        const e = window.innerWidth;
                        e < 1300 && e > 800 ? this.setState({
                            offSet: 2,
                            isNarrowScreen: !1
                        }) : e < 800 ? this.setState({
                            isNarrowScreen: !0
                        }) : e >= 1300 && this.setState({
                            offSet: 3,
                            isNarrowScreen: !1
                        })
                    }), Zn(this, "getPreviousPaperSet", () => {
                        const {
                            offSet: e,
                            currentIndex: t
                        } = this.state;
                        t < 0 && this.setState({
                            currentIndex: 0
                        }), 0 !== t && this.setState({
                            currentIndex: t - e
                        }), this.setState({
                            animationClassName: "carousel-left"
                        })
                    }), Zn(this, "getNextPaperSet", () => {
                        const {
                            offSet: e,
                            currentIndex: t
                        } = this.state, {
                            paperDetail: {
                                relatedPapers: r
                            }
                        } = this.props;
                        t === r.size - 1 || t < 0 ? this.setState({
                            currentIndex: 0
                        }) : this.setState({
                            currentIndex: t + e
                        }), this.setState({
                            animationClassName: "carousel-right"
                        })
                    }), Zn(this, "renderIndicators", () => {
                        const {
                            paperDetail: {
                                relatedPapers: e
                            }
                        } = this.props, {
                            currentIndex: t,
                            offSet: r
                        } = this.state, a = ei(e, r), n = a.size;
                        return c.a.createElement("ul", {
                            className: "carousel-indicators"
                        }, a.map((e, a) => c.a.createElement($n, {
                            key: a,
                            currentIndex: t,
                            indicatorIndex: a,
                            offSet: r,
                            paperCount: n
                        })))
                    }), Zn(this, "onShowMoreClick", () => {
                        const {
                            showMoreToggle: e
                        } = this.state;
                        this.setState({
                            showMoreToggle: !e
                        })
                    }), Zn(this, "setCarouselContainer", e => {
                        this._carouselContainerRef = e
                    }), this.state = {
                        paperFetchCount: 10,
                        paperRecommendationType: "relatedPapers",
                        currentIndex: 0,
                        offSet: 3,
                        screenWidth: null,
                        animationClassName: "",
                        isNarrowScreen: !1,
                        smallScreenOffSet: 2,
                        showMoreToggle: !0,
                        heightPx: null
                    }
                }
                componentDidMount() {
                    He.a.requestAnimationFrame(() => {
                        this.setState({
                            screenWidth: window.innerWidth
                        })
                    }), window.matchMedia("(max-width: 800px)").onchange = this.selectSizeRender, window.matchMedia("(min-width: 800px) and (max-width: 1300px)").onchange = this.selectSizeRender, window.matchMedia("(min-width: 1300px)").onchange = this.selectSizeRender, this.selectSizeRender()
                }
                componentDidUpdate() {
                    if (!this.props.paperDetail.isLoadingRelatedPapers && !this.state.heightPx) {
                        const e = this._carouselContainerRef ? this._carouselContainerRef.clientHeight : 0;
                        this.setState({
                            heightPx: e
                        })
                    }
                }
                renderSimilarPapers() {
                    const {
                        offSet: e,
                        currentIndex: t,
                        paperFetchCount: r,
                        paperRecommendationType: a,
                        animationClassName: n,
                        isNarrowScreen: i,
                        smallScreenOffSet: s,
                        showMoreToggle: o
                    } = this.state;
                    return c.a.createElement(Xn.CSSTransitionGroup, {
                        transitionName: n,
                        transitionEnterTimeout: 600,
                        transitionLeaveTimeout: 600
                    }, c.a.createElement(Kn, {
                        key: t,
                        paperDetail: this.props.paperDetail,
                        currentIndex: t,
                        paperFetchCount: r,
                        paperRecommendationType: a,
                        offSet: e,
                        isNarrowScreen: i,
                        smallScreenOffSet: s,
                        showMoreToggle: o
                    }))
                }
                showLeftArrow() {
                    const {
                        currentIndex: e
                    } = this.state;
                    return !(e <= 0)
                }
                showRightArrow() {
                    const {
                        currentIndex: e,
                        offSet: t
                    } = this.state, {
                        paperDetail: {
                            relatedPapers: r
                        }
                    } = this.props, a = ei(r, t);
                    return !(e + t >= a.size) && !(e === a.size - 1)
                }
                render() {
                    const {
                        showMoreToggle: e,
                        smallScreenOffSet: t,
                        isNarrowScreen: r,
                        offSet: a,
                        currentIndex: n,
                        heightPx: i
                    } = this.state, {
                        paperDetail: {
                            relatedPapers: s
                        }
                    } = this.props, o = s.size, l = i ? i / 2 - 20 + "px" : 0, p = {
                        "aria-label": Object(F.c)(e => e.paperDetail.relatedPapers.prevPapers)
                    }, u = {
                        "aria-label": Object(F.c)(e => e.paperDetail.relatedPapers.nextPapers)
                    }, d = ei(s, a).size;
                    return c.a.createElement("div", {
                        ref: this.setCarouselContainer,
                        className: "carousel-container"
                    }, !r && c.a.createElement("p", {
                        "aria-atomic": "true",
                        "aria-live": "polite",
                        className: "screen-reader-only"
                    }, Object(F.c)(e => e.paperDetail.relatedPapers.papersShowing, n + 1, n + a, d)), !r && this.renderIndicators(), this.showLeftArrow() && !r ? c.a.createElement("div", {
                        className: "carousal-arrow",
                        style: {
                            top: l,
                            left: "-2%"
                        }
                    }, c.a.createElement(qn.a, {
                        ariaProps: p,
                        direction: qn.a.Direction.LEFT,
                        onClick: this.getPreviousPaperSet
                    })) : null, c.a.createElement("div", {
                        className: "content-wrapper"
                    }, this.renderSimilarPapers()), this.showRightArrow() && !r ? c.a.createElement("div", {
                        className: "carousal-arrow",
                        style: {
                            top: l,
                            right: "-2%"
                        }
                    }, c.a.createElement(qn.a, {
                        ariaProps: u,
                        direction: qn.a.Direction.RIGHT,
                        onClick: this.getNextPaperSet
                    })) : null, r && c.a.createElement("div", {
                        className: "narrow-view-footer-container"
                    }, c.a.createElement("div", {
                        className: "show-more"
                    }, c.a.createElement(qn.a, {
                        direction: e ? qn.a.Direction.DOWN : qn.a.Direction.UP,
                        onClick: this.onShowMoreClick,
                        "aria-label": e ? Object(F.c)(e => e.paperDetail.relatedPapers.narrowViewToggle.showMoreAriaLabel) : Object(F.c)(e => e.paperDetail.relatedPapers.narrowViewToggle.showLessAriaLabel),
                        className: "carousal-arrow"
                    }), c.a.createElement("p", {
                        className: "toggle-text"
                    }, e ? Object(F.c)(e => e.paperDetail.relatedPapers.narrowViewToggle.ShowMoreToggleText) : Object(F.c)(e => e.paperDetail.relatedPapers.narrowViewToggle.ShowLessToggleText))), c.a.createElement("div", {
                        "aria-live": "polite",
                        "aria-atomic": "true",
                        className: "number-indicator"
                    }, e ? t.toLocaleString() : o.toLocaleString(), "/", o.toLocaleString(), c.a.createElement("span", {
                        className: "screen-reader-only"
                    }, Object(F.c)(e => e.paperDetail.relatedPapers.narrowViewToggle.relatedPapers)))))
                }
            }

            function ei(e, t) {
                const r = e.size % t;
                return e.slice(0, e.size - r)
            }
            Zn(Jn, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired
            });
            class ti extends l.PureComponent {
                render() {
                    const {
                        isMobile: e
                    } = this.context.envInfo, {
                        paperDetail: t
                    } = this.props;
                    return l.createElement(P, {
                        id: "related-papers",
                        navLabel: Object(F.c)(e => e.paperDetail.tabLabels.similarPapers),
                        className: w()("similar-papers", {
                            "similar-papers--mobile": e
                        })
                    }, l.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: Object(F.c)(e => e.paperDetail.sectionTitles.similarPapers)
                    }, l.createElement("h2", {
                        className: "card-header-title__carousel"
                    }, Object(F.c)(e => e.paperDetail.sectionTitles.similarPapers)), l.createElement(Jn, {
                        paperDetail: t
                    })))
                }
            }! function(e, t, r) {
                t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r
            }(ti, "contextTypes", {
                envInfo: O.a.instanceOf(b.a).isRequired
            });
            var ri = Object(Ke.b)(K.a.PaperDetail.RelatedPapers)(ti),
                ai = r(467),
                ni = r(41),
                ii = r(45),
                si = r(486);

            function oi(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class li extends l.Component {
                constructor() {
                    super(...arguments), oi(this, "state", {
                        isVisible: !!Object(jt.a)() && 0 !== window.scrollY
                    }), oi(this, "onScroll", () => {
                        this.setState({
                            isVisible: 0 !== window.scrollY
                        })
                    })
                }
                componentDidMount() {
                    He.a.listenForScroll(this.onScroll)
                }
                componentWillUnmount() {
                    He.a.stopListeningForScroll(this.onScroll)
                }
                render() {
                    const {
                        isVisible: e
                    } = this.state;
                    return l.createElement(N.default, {
                        className: w()(["button__scroll-top", {
                            "button__scroll-top--hidden": !e
                        }]),
                        type: "secondary",
                        icon: l.createElement(ue.a, {
                            icon: "arrow-up",
                            width: "16",
                            height: "24"
                        }),
                        iconPosition: "up",
                        onClick: () => {
                            window.scrollTo({
                                top: 0,
                                left: 0,
                                behavior: "smooth"
                            })
                        },
                        ariaProps: {
                            "aria-label": Object(F.c)(e => e.general.scrollToTop)
                        }
                    })
                }
            }
            var ci = r(23),
                pi = r.n(ci);

            function ui(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function di(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            class hi extends c.a.PureComponent {
                constructor() {
                    super(...arguments), di(this, "trackMetadataClick", (e, t) => {
                        const r = this.props.paperDetail.paper.id,
                            a = K.a.getIn(K.a.PaperDetail.Header, e);
                        Object(ct.a)(xt.a.create(a, {
                            paperId: r,
                            clickedEntityId: t
                        }))
                    }), di(this, "trackActionClick", (e, t) => {
                        const r = this.props.paperDetail.paper.id,
                            a = K.a.getIn(K.a.PaperDetail.Header, "Action", e),
                            n = ot.a.recursive(t, {
                                paperId: r
                            });
                        Object(ct.a)(xt.a.create(a, n))
                    }), di(this, "renderCards", () => {
                        const {
                            envInfo: {
                                isMobile: e
                            }
                        } = this.context, {
                            paperDetail: t,
                            paperDetail: {
                                paper: r
                            }
                        } = this.props, a = r.citationStats.numCitations > 0, n = t.citedPapers.totalCitations > 0, i = this.context.weblabStore.isFeatureEnabled(at.b.PDPReferencesFilterbar);
                        return c.a.createElement(c.a.Fragment, null, c.a.createElement("div", {
                            className: w()("centered-max-width-content", "paper-detail-page__cards", {
                                "paper-detail-page__cards--mobile": e
                            })
                        }, c.a.createElement(p, null, c.a.createElement(Nt, {
                            paperDetail: t
                        }), i ? c.a.createElement(c.a.Fragment, null, a && c.a.createElement(ft, {
                            paperDetail: t,
                            citationType: L.a.CITING_PAPERS
                        }), n && c.a.createElement(ft, {
                            paperDetail: t,
                            citationType: L.a.CITED_PAPERS
                        })) : c.a.createElement(c.a.Fragment, null, a && c.a.createElement(Ot, {
                            paperDetail: t
                        }), $a.hasContent({
                            paperDetail: t
                        }) && c.a.createElement($a, {
                            paperDetail: t
                        })), c.a.createElement(Wt.a, {
                            feature: at.b.SimilarPapersPdp
                        }, c.a.createElement(Gt.a, null, c.a.createElement(ri, {
                            paperDetail: t
                        }))))))
                    }), this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? ui(Object(r), !0).forEach((function(t) {
                                di(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : ui(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({
                        isInitialRender: !0
                    }, this.getStateFromWeblabStore()), this.context.weblabStore.registerComponent(this, () => {
                        this.setState(this.getStateFromWeblabStore())
                    })
                }
                componentDidMount() {
                    this.setState({
                        isInitialRender: !1
                    })
                }
                getStateFromWeblabStore() {
                    const e = this.context.weblabStore.getVariation(Qe.b.ABTestMock.KEY);
                    if (this.state && e === this.state.mockVariation) return {
                        heapGroup: this.state.heapGroup,
                        mockVariation: this.state.mockVariation
                    };
                    let t = 1;
                    /^test_\d+$/.test(e) && (t = parseFloat(e.replace("test_", "")) / 100);
                    return {
                        heapGroup: Math.random() <= t ? e : null,
                        mockVariation: e
                    }
                }
                getRelatedPapersQuery(e) {
                    const {
                        paper: {
                            entities: t
                        }
                    } = e;
                    return t.isEmpty() ? null : t.first().name
                }
                renderAuthorList() {
                    const {
                        paper: e
                    } = this.props.paperDetail, {
                        authors: t
                    } = e;
                    return t.isEmpty() ? null : c.a.createElement(ha, {
                        paper: e,
                        onClick: this.trackMetadataClick,
                        isFreshPDP: !0,
                        maxAuthors: 3
                    })
                }
                renderVenueAndYearListItem() {
                    var e;
                    const {
                        paper: t
                    } = this.props.paperDetail, {
                        pubDate: r,
                        year: a,
                        venue: n
                    } = t, i = !!!(null != (e = t) && null != (e = e.journal) ? e.name : e) && n && "string" == typeof n.text && n.text.length > 0, s = Object(ne.g)(a), o = Object(ne.f)(r), l = s || o;
                    let p, u;
                    if (i && l ? p = c.a.createElement(wn.a, {
                            paper: t,
                            stripYear: !0,
                            textLength: 30
                        }) : i && !l && (p = c.a.createElement(wn.a, {
                            paper: t,
                            stripYear: !1,
                            textLength: 30
                        })), u = o ? c.a.createElement(Vr.c, {
                            field: r
                        }) : a && "string" == typeof a.text ? c.a.createElement(Vr.c, {
                            field: a
                        }) : a, l || i) return c.a.createElement("span", {
                        key: "year-and-venue",
                        "data-test-id": "year-and-venue"
                    }, Object(F.c)(e => e.paper.meta.published), i ? " in " : null, i ? c.a.createElement("span", {
                        "data-test-id": "venue-metadata"
                    }, p) : null, l ? " " : null, l ? c.a.createElement("span", {
                        "data-test-id": "paper-year"
                    }, u) : null)
                }
                renderFieldOfStudy() {
                    const {
                        paper: e
                    } = this.props.paperDetail;
                    return e.fieldsOfStudy && e.fieldsOfStudy.size ? c.a.createElement(Pn, {
                        paper: e,
                        "data-heap-id": "paper-meta-fos"
                    }) : null
                }
                renderJournal() {
                    var e;
                    const {
                        paper: t
                    } = this.props.paperDetail;
                    return (null != (e = t) && null != (e = e.journal) ? e.name : e) ? c.a.createElement("span", {
                        "data-heap-id": "paper-meta-journal"
                    }, t.journal.name) : null
                }
                render() {
                    const {
                        paperDetail: e,
                        paperDetail: {
                            paper: t,
                            entitlement: r,
                            skipperExperiments: a
                        }
                    } = this.props, {
                        cookieJar: n,
                        envInfo: {
                            isMobile: i
                        },
                        history: s
                    } = this.context, {
                        heapGroup: o,
                        isInitialRender: l
                    } = this.state, p = (() => {
                        if ("claim" in pi.a.parse(s.location.search.replace(/^\?/, ""))) return !0;
                        const e = n.getCookie(ni.a);
                        if (e) {
                            const [r, a] = e.split(",");
                            if (a === t.id) return !0
                        }
                        return !1
                    })(), u = this.renderCards();
                    return c.a.createElement("div", {
                        className: "fresh-paper-detail-page paper-detail-cards",
                        role: "main",
                        id: "main-content",
                        "data-heap-id": Qe.b.ABTestMock.KEY,
                        "data-heap-group": o
                    }, c.a.createElement("div", {
                        className: "fresh-paper-detail-page__above-the-fold"
                    }, c.a.createElement("div", {
                        className: "centered-max-width-content"
                    }, p && c.a.createElement(Wt.a, {
                        feature: e => e.AuthorClaimOnPDP
                    }, c.a.createElement(Gt.a, null, c.a.createElement(vn, {
                        paperDetail: e
                    }))), c.a.createElement(te.a, {
                        layout: "row",
                        layoutMed: "column",
                        position: "relative"
                    }, c.a.createElement(Dt.a, {
                        className: "flex-item__left-column",
                        width: "66"
                    }, c.a.createElement("div", {
                        className: i ? "fresh-paper-detail-page__header__mobile" : "fresh-paper-detail-page__header",
                        "data-test-id": "paper-detail-page-header"
                    }, c.a.createElement(ua.a, {
                        paper: t,
                        onClick: this.trackMetadataClick,
                        isFreshPDP: !0,
                        maxAuthors: 3,
                        isPDPMeta: !0,
                        className: "paper-detail__paper-meta-top"
                    }), c.a.createElement("h1", {
                        "data-test-id": "paper-detail-title"
                    }, t.title.text), c.a.createElement("pre", {
                        className: "bibtex-citation",
                        "data-nosnippet": !0
                    }, Object(si.a)(t)), c.a.createElement(Sn, {
                        childList: [this.renderAuthorList(), this.renderVenueAndYearListItem(), this.renderFieldOfStudy(), this.renderJournal()]
                    }), c.a.createElement(St, {
                        paper: t,
                        skipperExperiments: a
                    }), t.paperAbstract.text ? c.a.createElement(fn, {
                        experiment: Qe.b.PaperHighlightAbstractV2,
                        eligibilityTest: () => e.skipperExperiments.memberOf("paper-abstract-highlight-v2")
                    }, c.a.createElement(Za.a, null, c.a.createElement("div", {
                        className: "fresh-paper-detail-page__abstract",
                        "data-test-id": "abstract-text"
                    }, c.a.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: "Abstract"
                    }), c.a.createElement(ai.a, {
                        className: "abstract__text text--preline",
                        text: t.paperAbstract.text,
                        limit: i ? 200 : 500
                    }))), c.a.createElement(hn.a, {
                        variation: Qe.b.PaperHighlightAbstractV2.Variation.HIGHLIGHTED_ABSTRACT_DEFAULT_TOGGLE_OFF
                    }, c.a.createElement(pa, {
                        paperDetail: e,
                        isMobile: i
                    }))) : null, c.a.createElement(Br, {
                        paper: t,
                        entitlement: r,
                        trackActionClick: this.trackActionClick,
                        isPdp: !0
                    }))), c.a.createElement(Dt.a, {
                        className: "flex-item__right-column",
                        width: "33"
                    }, c.a.createElement(Mn.a, {
                        className: "paper-detail-page__share-social-options",
                        title: t.title.text,
                        corpusId: t.corpusId,
                        label: Object(F.c)(e => e.socialShareOptions.shareButtonLabel.sharePaper)
                    }), c.a.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: "Scorecard"
                    }), c.a.createElement(Fn, {
                        paperDetail: e
                    }))))), !l && c.a.createElement("div", {
                        className: "paper-detail-page__paper-nav",
                        key: "papernav"
                    }, c.a.createElement(ya, null)), u, l && c.a.createElement("div", {
                        className: "paper-detail-page__paper-nav",
                        key: "papernav"
                    }, c.a.createElement(ya, null)), c.a.createElement(li, null))
                }
            }
            di(hi, "contextTypes", {
                cookieJar: O.a.instanceOf(ii.b).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                paperStore: O.a.instanceOf(qa.b).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            });
            var mi = r(24),
                bi = r(582),
                fi = r(602);

            function gi(e, t) {
                var r = Object.keys(e);
                if (Object.getOwnPropertySymbols) {
                    var a = Object.getOwnPropertySymbols(e);
                    t && (a = a.filter((function(t) {
                        return Object.getOwnPropertyDescriptor(e, t).enumerable
                    }))), r.push.apply(r, a)
                }
                return r
            }

            function yi(e, t, r) {
                return t in e ? Object.defineProperty(e, t, {
                    value: r,
                    enumerable: !0,
                    configurable: !0,
                    writable: !0
                }) : e[t] = r, e
            }
            r.d(t, "default", (function() {
                return Oi
            }));
            class Oi extends c.a.Component {
                constructor() {
                    super(...arguments), yi(this, "onAuthStoreChange", () => {
                        const {
                            dispatcher: e
                        } = this.context;
                        setTimeout(async () => {
                            const t = await this.apiFetchPaperDetail(),
                                r = Object(bi.a)(t);
                            r && e.dispatch(r)
                        }, 5)
                    }), yi(this, "getStateFromPaperStore", () => ({
                        loading: this.context.paperStore.isLoading(),
                        paperDetail: this.context.paperStore.getPaperDetail()
                    })), yi(this, "apiFetchPaperDetail", () => {
                        const {
                            paperDetail: e
                        } = this.state, t = e.paper.id;
                        return this.context.api.fetchPaperDetail({
                            paperId: t,
                            requireSlug: !1
                        })
                    }), yi(this, "fetchPaperLite", async e => {
                        var t;
                        const {
                            api: r
                        } = this.context, a = ((null != (t = await r.fetchPapersByIds({
                            paperIds: [e],
                            model: "PaperLite"
                        })) && null != (t = t.resultData) ? t.paperLites : t) || []).map(e => Object(_e.c)(e)), [n] = a;
                        return oa()(n, `API didn't return a paper to paperId="${e}"`), n
                    }), yi(this, "openOrganizePapersShelf", async () => {
                        const {
                            dispatcher: e
                        } = this.context, {
                            paperId: t
                        } = this.props, r = await this.fetchPaperLite(t), a = Object(Cr.d)({
                            paperId: t,
                            paperTitle: r.title
                        });
                        e.dispatch(a)
                    }), yi(this, "fetchLibraryFolders", () => {
                        const {
                            api: e,
                            libraryFolderStore: t
                        } = this.context;
                        return t.isUninitialized() ? e.getLibraryFolders() : Promise.resolve()
                    }), yi(this, "handleSaveToLibraryWithShelf", async () => {
                        const {
                            api: e,
                            messageStore: t,
                            dispatcher: r
                        } = this.context, {
                            paperId: a
                        } = this.props, n = await this.fetchPaperLite(a);
                        this.fetchLibraryFolders().then(() => {
                            const i = Object(xr.a)({
                                paperId: a,
                                paperTitle: n.title,
                                sourceType: wr.c
                            });
                            e.createLibraryEntryBulk(i).catch(e => {
                                t.addError(Object(F.c)(e => e.library.saveToLibraryShelf.errorMessage)), Tr.a.error(e)
                            }), r.dispatch(Object(Cr.f)({
                                paper: n,
                                selectedFolders: Et.b.Set()
                            }))
                        }).catch(e => {
                            Object(We.default)("library", `failed to open Save To Library shelf for paperId="${a}"]`, e);
                            const r = Object(F.c)(e => e.library.message.error.header),
                                n = Object(F.c)(e => e.library.message.error.body);
                            t.addError(n, r)
                        })
                    }), yi(this, "saveToLibraryNew", () => {
                        const {
                            authStore: e,
                            dispatcher: t
                        } = this.context;
                        e.ensureLogin({
                            dispatcher: t,
                            location: _r.g.pdpLibrary
                        }).then(async () => {
                            await this.fetchLibraryFolders(), this.getStateFromLibraryFolderStore().isInLibrary ? this.openOrganizePapersShelf() : this.handleSaveToLibraryWithShelf()
                        }, e => {
                            Tr.a.warn(e)
                        })
                    });
                    const {
                        authStore: e,
                        libraryFolderStore: t,
                        paperStore: r
                    } = this.context;
                    this.state = function(e) {
                        for (var t = 1; t < arguments.length; t++) {
                            var r = null != arguments[t] ? arguments[t] : {};
                            t % 2 ? gi(Object(r), !0).forEach((function(t) {
                                yi(e, t, r[t])
                            })) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(r)) : gi(Object(r)).forEach((function(t) {
                                Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(r, t))
                            }))
                        }
                        return e
                    }({}, this.getStateFromLibraryFolderStore(), {}, this.getStateFromPaperStore()), e.registerComponent(this, this.onAuthStoreChange), t.registerComponent(this, () => {
                        this.setState(this.getStateFromLibraryFolderStore())
                    }), r.registerComponent(this, () => {
                        this.setState(this.getStateFromPaperStore())
                    })
                }
                componentDidMount() {
                    const {
                        history: e
                    } = this.context;
                    Object(mi.r)(e.location.search).save_to_library && this.saveToLibraryNew()
                }
                getStateFromLibraryFolderStore() {
                    const {
                        libraryFolderStore: e
                    } = this.context, {
                        paperId: t
                    } = this.props;
                    return {
                        isInLibrary: e.isPaperInLibrary(t)
                    }
                }
                render() {
                    const {
                        paperDetail: e,
                        loading: t
                    } = this.state, r = !t && c.a.createElement(ht, {
                        target: K.a.PaperDetail.SCROLL_LANDMARKS,
                        section: "Footer"
                    }, c.a.createElement(a.a, null));
                    return c.a.createElement(i.a, {
                        heapProps: {
                            "page-type": fi.b,
                            "page-paper-id": e.paper.id
                        }
                    }, c.a.createElement(o.a, {
                        key: e.paper.id,
                        className: this.context.envInfo.isMobile ? "detail" : "paper",
                        footer: r
                    }, t ? c.a.createElement("div", {
                        className: "container"
                    }, c.a.createElement("div", {
                        className: "loading-controls flex-row-vcenter",
                        role: "main",
                        id: "main-content"
                    }, c.a.createElement(s.a, {
                        testId: "paper-details-loading"
                    }), " Loading")) : c.a.createElement(c.a.Fragment, null, c.a.createElement(hi, {
                        paperDetail: e
                    }), c.a.createElement(n.b, {
                        paperDetail: e
                    }))))
                }
            }
            yi(Oi, "contextTypes", {
                api: O.a.instanceOf(m.a).isRequired,
                authStore: O.a.instanceOf(jr.a).isRequired,
                dispatcher: O.a.instanceOf(g.a).isRequired,
                envInfo: O.a.instanceOf(b.a).isRequired,
                history: O.a.instanceOf(X.a).isRequired,
                libraryFolderStore: O.a.instanceOf(kr.a).isRequired,
                messageStore: O.a.instanceOf(Ir.a).isRequired,
                paperStore: O.a.instanceOf(qa.b).isRequired,
                weblabStore: O.a.instanceOf($e.b).isRequired
            })
        }
    }
]);
//# sourceMappingURL=bundle-58.js.map