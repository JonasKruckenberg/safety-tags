//! Minimal stable-Rust emulation of RFC 3842 "safety tags" via linker poison
//! symbols + name resolution, with RFC-style attribute syntax at call sites.
//!
//! # Attributes
//!
//! - `#[safety::requires(Tag = "what callers must uphold", ...)]` on an
//!   `unsafe fn`:
//!   * moves the real body into a hidden associated fn `__safety_<hash>`
//!     (hash of the full sorted tag *name* set — reasons are documentation
//!     and deliberately not hashed) on an empty enum that shares the
//!     function's name; since `use` imports every namespace of a name, the
//!     hidden entry point resolves wherever the public fn does,
//!   * injects one body-local `fn __safety_delegates_<Tag>() {}` marker per
//!     tag into the real body (this is what makes `delegate(...)` resolve
//!     only inside functions that themselves require the tag),
//!   * re-emits the original name as an `#[inline(always)]` wrapper that
//!     trips a post-monomorphization const-eval panic and then forwards.
//!     The panic lives in an associated const behind a generic parameter,
//!     so it is only evaluated when the wrapper is instantiated — and the
//!     wrapper, being `#[inline(always)]`, is only instantiated if someone
//!     actually calls the public (unchecked) name. A clean program never
//!     evaluates it; an unchecked call fails `cargo build` of the crate
//!     containing the call (no final link needed), with an error message
//!     listing every required tag and its reason. (A plain
//!     `compile_error!()` cannot work here: it fires unconditionally at
//!     macro expansion time, long before anyone knows the call sites.)
//!   * appends an auto-generated `# Safety` rustdoc section listing every
//!     tag and its reason (this also satisfies
//!     `clippy::missing_safety_doc`),
//!   * expands any `#[safety::checked(...)]` attributes inside its body.
//!
//! - `#[safety::checks]` on any fn: expands `#[safety::checked(...)]`
//!   attributes inside the body. Needed because stable Rust has no attribute
//!   macros on expressions: the *enclosing item* macro consumes the inner
//!   attributes before rustc ever parses them (syn accepts expression
//!   attributes even though stable rustc does not). Forgetting it is safe:
//!   a leftover `#[safety::checked]` is a hard error on stable, never a
//!   silent no-op.
//!
//! - `#[safety::checked(TagA = "why it holds here", delegate(TagB = "why
//!   forwarding it is sound"))]` on an `unsafe { f(x) }` block, a direct
//!   call, or a `let` statement (inside a `requires`/`checks` fn):
//!   * derives the hidden name from the union of discharged + delegated
//!     tag names and rewrites the call to it (wrong/missing tags simply
//!     fail name resolution at compile time),
//!   * requires a reason string per tag, discharged *and* delegated — the
//!     per-call-site safety comment — and embeds all reasons into the
//!     expansion as `const _: &[(&str, &str)]` so they remain
//!     machine-readable for tooling,
//!   * emits `let _ = __safety_delegates_<Tag>;` for each delegated tag,
//!     which only compiles inside a function carrying
//!     `#[safety::requires(Tag = "...")]`.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    visit_mut::{self, VisitMut},
    Attribute, Expr, FnArg, Ident, ItemFn, LitStr, Pat, Stmt, Token,
};

// ---------------------------------------------------------------------------
// Tag specs and tag-set hashing (stable, order/duplicate-insensitive)
// ---------------------------------------------------------------------------

/// `Tag = "reason"` (reason optional at parse time; each attribute decides
/// where it is mandatory).
struct TagSpec {
    name: Ident,
    reason: Option<LitStr>,
}

impl Parse for TagSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        let reason = if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            Some(input.parse::<LitStr>()?)
        } else {
            None
        };
        Ok(TagSpec { name, reason })
    }
}

impl TagSpec {
    fn require_reason(&self, what: &str) -> syn::Result<&LitStr> {
        self.reason.as_ref().ok_or_else(|| {
            syn::Error::new(
                self.name.span(),
                format!(
                    "safety tag `{}` is missing its reason string: write `{} = \"{}\"`",
                    self.name, self.name, what
                ),
            )
        })
    }
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Hash of the *set* of tag names. Reasons are intentionally excluded:
/// rewording documentation must not break callers — if a tag's meaning
/// changes, rename the tag (that is the semver story). Note matching is
/// purely textual: proc macros cannot do name resolution, so tags must be
/// written as bare identifiers on both sides.
fn tagset_hash(tags: &[Ident]) -> String {
    let mut names: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
    names.sort();
    names.dedup();
    format!("{:016x}", fnv1a(&names.join(",")))
}

/// Name of the hidden associated entry point, parameterized only by the tag
/// set. It lives in `impl <fn_name> { .. }` on an empty enum that shadows the
/// public fn in the *type* namespace, so `use lib::f;` (or `use lib::f as g;`)
/// brings both along and `f::__safety_<hash>(..)` resolves wherever `f(..)`
/// did.
fn hidden_ident(tags: &[Ident]) -> Ident {
    format_ident!("__safety_{}", tagset_hash(tags))
}

fn marker_ident(tag: &Ident) -> Ident {
    format_ident!("__safety_delegates_{}", tag, span = tag.span())
}

// ---------------------------------------------------------------------------
// #[safety::checked(Tag = "...", ..., delegate(Tag, ...))] — inner attribute,
// consumed by `requires`/`checks` on the enclosing fn.
// ---------------------------------------------------------------------------

struct CheckedArgs {
    discharged: Vec<TagSpec>,
    delegated: Vec<TagSpec>,
}

impl Parse for CheckedArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut discharged: Vec<TagSpec> = Vec::new();
        let mut delegated: Vec<TagSpec> = Vec::new();

        while !input.is_empty() {
            if input.peek(Ident) && input.peek2(syn::token::Paren) {
                let kw: Ident = input.parse()?;
                if kw != "delegate" {
                    return Err(syn::Error::new(
                        kw.span(),
                        "expected `Tag = \"reason\"` or `delegate(Tag = \"reason\", ...)`",
                    ));
                }
                let content;
                syn::parenthesized!(content in input);
                for spec in Punctuated::<TagSpec, Token![,]>::parse_terminated(&content)? {
                    spec.require_reason("why passing this obligation to our own callers is sound")?;
                    delegated.push(spec);
                }
            } else {
                let spec: TagSpec = input.parse()?;
                spec.require_reason("why this invariant holds at this call site")?;
                discharged.push(spec);
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        if discharged.is_empty() && delegated.is_empty() {
            return Err(input.error("expected at least one tag"));
        }
        Ok(CheckedArgs {
            discharged,
            delegated,
        })
    }
}

/// `#[checked(...)]` / `#[safety::checked(...)]`
fn is_checked_attr(attr: &Attribute) -> bool {
    let segs = &attr.path().segments;
    match segs.len() {
        1 => segs[0].ident == "checked",
        2 => segs[0].ident == "safety" && segs[1].ident == "checked",
        _ => false,
    }
}

fn take_checked_attr(attrs: &mut Vec<Attribute>) -> Option<Attribute> {
    let pos = attrs.iter().position(is_checked_attr)?;
    Some(attrs.remove(pos))
}

/// Rewrite the single (optionally `unsafe { .. }`-wrapped) call expression
/// so its final path gains a `::__safety_<hash>` segment.
fn rewrite_call(expr: &mut Expr, suffix: &str) -> syn::Result<()> {
    match expr {
        Expr::Unsafe(u) => {
            if u.block.stmts.len() != 1 {
                return Err(syn::Error::new_spanned(
                    &*u,
                    "#[safety::checked] requires the unsafe block to contain exactly one call",
                ));
            }
            match &mut u.block.stmts[0] {
                Stmt::Expr(inner, _) => rewrite_call(inner, suffix),
                other => Err(syn::Error::new_spanned(
                    other,
                    "#[safety::checked] requires the unsafe block to contain exactly one call",
                )),
            }
        }
        Expr::Call(call) => match &mut *call.func {
            Expr::Path(p) => {
                let Some(last) = p.path.segments.last_mut() else {
                    return Err(syn::Error::new(
                        proc_macro2::Span::call_site(),
                        "empty call path",
                    ));
                };
                // `f(..)` -> `f::__safety_<hash>(..)`; a turbofish moves onto
                // the associated fn: `f::<T>(..)` -> `f::__safety_<hash>::<T>(..)`.
                let generics = std::mem::replace(&mut last.arguments, syn::PathArguments::None);
                let hidden = format_ident!("__safety_{}", suffix, span = last.ident.span());
                p.path.segments.push(syn::PathSegment {
                    ident: hidden,
                    arguments: generics,
                });
                Ok(())
            }
            other => Err(syn::Error::new_spanned(
                other,
                "#[safety::checked] only supports direct path calls like `f(..)` or `path::to::f(..)`",
            )),
        },
        Expr::MethodCall(m) => Err(syn::Error::new_spanned(
            m,
            "#[safety::checked] does not support method-call syntax; use a path call",
        )),
        other => Err(syn::Error::new_spanned(
            other,
            "#[safety::checked] expects a single function call, optionally wrapped in `unsafe { .. }`",
        )),
    }
}

/// Apply one parsed `#[checked]` attribute to the expression it annotated.
fn expand_checked(attr: Attribute, expr: &mut Expr) -> syn::Result<()> {
    let CheckedArgs {
        discharged,
        delegated,
    } = attr.parse_args()?;

    // The hidden name encodes the *union*: discharged + delegated must
    // together cover exactly the callee's required tag set, or this fails
    // to resolve (compile error at the call site).
    let union: Vec<Ident> = discharged
        .iter()
        .chain(delegated.iter())
        .map(|s| s.name.clone())
        .collect();
    rewrite_call(expr, &tagset_hash(&union))?;

    // Each delegated tag must resolve against a body-local marker injected
    // by #[safety::requires(Tag = "...")] on the *enclosing* function.
    let marker_refs: Vec<proc_macro2::TokenStream> = delegated
        .iter()
        .map(|s| {
            let m = marker_ident(&s.name);
            quote! { let _ = #m; }
        })
        .collect();

    // Embed the per-call-site justifications as data so external tooling
    // (e.g. a future clippy-style checker, or a section-scanner) can read
    // them back out of the expansion.
    let justifications: Vec<proc_macro2::TokenStream> = discharged
        .iter()
        .chain(delegated.iter())
        .map(|s| {
            let name = s.name.to_string();
            let reason = s.reason.as_ref().map(|r| r.value()).unwrap_or_default();
            quote! { (#name, #reason) }
        })
        .collect();

    let inner = std::mem::replace(expr, Expr::Verbatim(Default::default()));
    *expr = parse_quote!({
        const _: &[(&str, &str)] = &[ #(#justifications),* ];
        #(#marker_refs)*
        #inner
    });
    Ok(())
}

/// Walks a function body, consuming every `#[safety::checked(...)]` before
/// rustc can reject it (expression attributes are unstable, but they only
/// exist as tokens inside our input — syn parses them, we strip them).
struct CheckedExpander {
    error: Option<syn::Error>,
}

impl CheckedExpander {
    fn push(&mut self, e: syn::Error) {
        match &mut self.error {
            Some(prev) => prev.combine(e),
            None => self.error = Some(e),
        }
    }
}

impl VisitMut for CheckedExpander {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // The attribute may sit on the unsafe block or directly on the call;
        // statement-position attributes also land on the expression in syn.
        let taken = match expr {
            Expr::Unsafe(u) => take_checked_attr(&mut u.attrs),
            Expr::Call(c) => take_checked_attr(&mut c.attrs),
            Expr::MethodCall(m) => take_checked_attr(&mut m.attrs),
            _ => None,
        };
        if let Some(attr) = taken {
            if let Err(e) = expand_checked(attr, expr) {
                self.push(e);
            }
        }
        visit_mut::visit_expr_mut(self, expr);
    }

    // `#[safety::checked(...)] let v = unsafe { f(x) };` — the attribute is
    // on the `let`; apply it to the initializer expression.
    fn visit_local_mut(&mut self, local: &mut syn::Local) {
        if let Some(attr) = take_checked_attr(&mut local.attrs) {
            match &mut local.init {
                Some(init) => {
                    if let Err(e) = expand_checked(attr, &mut init.expr) {
                        self.push(e);
                    }
                }
                None => self.push(syn::Error::new_spanned(
                    &*local,
                    "#[safety::checked] on a `let` requires an initializer",
                )),
            }
        }
        visit_mut::visit_local_mut(self, local);
    }
}

fn expand_body_checks(func: &mut ItemFn) -> syn::Result<()> {
    let mut expander = CheckedExpander { error: None };
    expander.visit_block_mut(&mut func.block);
    match expander.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// #[safety::checks] — enables #[safety::checked] inside any fn body
// ---------------------------------------------------------------------------

#[proc_macro_attribute]
pub fn checks(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[safety::checks] takes no arguments",
        )
        .to_compile_error()
        .into();
    }
    let mut func = parse_macro_input!(item as ItemFn);
    match expand_body_checks(&mut func) {
        Ok(()) => quote!(#func).into(),
        Err(e) => e.to_compile_error().into(),
    }
}

// ---------------------------------------------------------------------------
// #[safety::requires(Tag = "...", ...)]
// ---------------------------------------------------------------------------

struct TagList(Vec<TagSpec>);

impl Parse for TagList {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let tags = Punctuated::<TagSpec, Token![,]>::parse_terminated(input)?;
        if tags.is_empty() {
            return Err(input
                .error("expected at least one safety tag: #[safety::requires(Tag = \"reason\")]"));
        }
        Ok(TagList(tags.into_iter().collect()))
    }
}

#[proc_macro_attribute]
pub fn requires(attr: TokenStream, item: TokenStream) -> TokenStream {
    let tags = parse_macro_input!(attr as TagList).0;
    let func = parse_macro_input!(item as ItemFn);

    match expand_requires(tags, func) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn func_type_params(f: &ItemFn) -> Vec<&Ident> {
    f.sig
        .generics
        .params
        .iter()
        .filter_map(|p| match p {
            syn::GenericParam::Type(t) => Some(&t.ident),
            _ => None,
        })
        .collect()
}

fn expand_requires(specs: Vec<TagSpec>, mut func: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    if func.sig.unsafety.is_none() {
        return Err(syn::Error::new_spanned(
            &func.sig.fn_token,
            "#[safety::requires] may only be applied to `unsafe fn`",
        ));
    }

    // Reasons are mandatory on the requiring side: they *define* the
    // invariant and become the auto-generated `# Safety` docs.
    for spec in &specs {
        spec.require_reason("what callers must uphold")?;
    }
    let tags: Vec<Ident> = specs.iter().map(|s| s.name.clone()).collect();

    // This fn may itself contain checked call sites (e.g. when delegating).
    expand_body_checks(&mut func)?;

    let public_name = func.sig.ident.clone();
    let hidden_name = hidden_ident(&tags);

    // --- forwarding arguments (simple ident patterns only, no `self`) ------
    let mut fwd_args = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Receiver(r) => {
                return Err(syn::Error::new_spanned(
                    r,
                    "this minimal implementation does not support methods (`self`)",
                ))
            }
            FnArg::Typed(pt) => match &*pt.pat {
                Pat::Ident(pi) => fwd_args.push(pi.ident.clone()),
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "this minimal implementation only supports plain identifier parameters",
                    ))
                }
            },
        }
    }

    // Explicit turbofish so forwarding works even when a type/const parameter
    // is not inferrable from the value arguments.
    let generic_args: Vec<proc_macro2::TokenStream> = func
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| match p {
            syn::GenericParam::Type(t) => {
                let id = &t.ident;
                Some(quote!(#id))
            }
            syn::GenericParam::Const(c) => {
                let id = &c.ident;
                Some(quote!(#id))
            }
            syn::GenericParam::Lifetime(_) => None,
        })
        .collect();
    let turbofish = if generic_args.is_empty() {
        quote!()
    } else {
        quote!(::<#(#generic_args),*>)
    };

    // --- violation message for the post-mono const-eval panic --------------
    let krate = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let mut violation_msg = format!(
        "SAFETY VIOLATION: unchecked call to `{krate}::{public_name}`. \
Discharge every tag with #[safety::checked(..)]:"
    );
    for spec in &specs {
        violation_msg.push_str(&format!(
            " [{}: {}]",
            spec.name,
            spec.reason.as_ref().map(|r| r.value()).unwrap_or_default()
        ));
    }

    // --- hidden entry point: assoc fn on a same-named empty enum ------------
    // The empty enum occupies only the type namespace, so it coexists with
    // the wrapper fn (value namespace) under one name, and any `use` of the
    // public fn imports both.
    let vis = func.vis.clone();
    let mut hidden_fn = func.clone();
    hidden_fn.vis = parse_quote!(pub); // reachability bounded by the enum's vis
    hidden_fn.sig.ident = hidden_name.clone();
    hidden_fn.attrs = vec![
        parse_quote!(#[doc(hidden)]),
        parse_quote!(#[allow(non_snake_case, clippy::missing_safety_doc)]),
    ];
    let markers: Vec<Stmt> = tags
        .iter()
        .map(|t| {
            let m = marker_ident(t);
            parse_quote! {
                #[allow(dead_code, non_snake_case)]
                fn #m() {}
            }
        })
        .collect();
    hidden_fn.block.stmts.splice(0..0, markers);

    let ns_enum = quote! {
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #vis enum #public_name {}

        impl #public_name {
            #hidden_fn

            /// Fallback tripwire for non-generic functions: the const is
            /// polymorphic over `__Safety`, and this helper is only
            /// *instantiated* (rustc's lazy boundary is generic-fn
            /// instantiation, not const lookup) when the wrapper is
            /// codegen'd — i.e. iff an unchecked call exists.
            #[doc(hidden)]
            pub fn __safety_violation<__Safety: ?Sized>() {
                struct __Poison<__Safety: ?Sized>(::core::marker::PhantomData<__Safety>);
                impl<__Safety: ?Sized> __Poison<__Safety> {
                    const TRIPPED: () = panic!(#violation_msg);
                }
                let _ = __Poison::<__Safety>::TRIPPED;
            }
        }
    };

    // --- public wrapper: tripwire + forward ----------------------------------
    let mut wrapper = func;
    // Auto-generated safety docs from the tag reasons. Appending after the
    // user's own docs; also satisfies clippy::missing_safety_doc.
    wrapper.attrs.push(parse_quote!(#[doc = ""]));
    wrapper.attrs.push(parse_quote!(#[doc = " # Safety"]));
    wrapper.attrs.push(parse_quote!(
        #[doc = " Callers must discharge each tag with `#[safety::checked(..)]`:"]
    ));
    for spec in &specs {
        let line = format!(
            " - **`{}`** — {}",
            spec.name,
            spec.reason.as_ref().map(|r| r.value()).unwrap_or_default()
        );
        wrapper.attrs.push(parse_quote!(#[doc = #line]));
    }
    // Tripwire selection. Rule learned the hard way: a const that is
    // MONOMORPHIC in its enclosing fn is evaluated eagerly when the
    // *defining* crate compiles, even if the fn is never instantiated; only
    // generic-dependent consts (and generic fn instantiation) are lazy.
    //  - Generic fn: embed the const directly in the wrapper, parameterized
    //    by the fn's own type params. Innermost instantiation above the
    //    error is then `f::<Args>` itself, so rustc's "while instantiating"
    //    note points at the offending CALL SITE.
    //  - Non-generic fn: call the per-fn generic helper on the namespace
    //    enum. Dormant for the same reason; the note points at the callee,
    //    but the panic message still names crate, fn, tags, and reasons.
    let type_params: Vec<&Ident> = func_type_params(&wrapper);
    let tripwire: proc_macro2::TokenStream = if type_params.is_empty() {
        quote!( #public_name::__safety_violation::<()>(); )
    } else {
        quote!(
            struct __Poison<__Safety: ?Sized>(::core::marker::PhantomData<__Safety>);
            impl<__Safety: ?Sized> __Poison<__Safety> {
                const TRIPPED: () = panic!(#violation_msg);
            }
            let _ = __Poison::<( #(#type_params,)* )>::TRIPPED;
        )
    };
    wrapper.attrs.push(parse_quote!(#[allow(unused_unsafe)]));
    wrapper.attrs.push(parse_quote!(#[inline(always)])); // critical: instantiated only if actually called
    wrapper.block = Box::new(parse_quote!({
        #tripwire
        unsafe { #public_name::#hidden_name #turbofish ( #(#fwd_args),* ) }
    }));

    Ok(quote! {
        #ns_enum
        #wrapper
    })
}
