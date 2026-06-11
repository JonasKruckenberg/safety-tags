// #[safety::checks] makes #[safety::checked(...)] usable inside this body.
// Forgetting it is fail-closed: a leftover #[safety::checked] is a hard
// compile error on stable, never a silent no-op.
#[safety::checks]
fn main() {
    let x: u32 = 0xDEAD_BEEF;
    let bytes = [1u8, 2, 3];

    // Fully discharged at this call site: union {ValidPtr, Init} matches the
    // callee's tag set, so this resolves to the hidden entry point.
    #[safety::checked(
        ValidPtr = "`&x` is a reference, hence non-null, aligned, valid for reads",
        Init = "`x` is a fully initialized local"
    )]
    let v = unsafe { read(&x) };

    // Calling the delegating wrapper: it requires only ValidPtr.
    #[safety::checked(ValidPtr = "pointer derives from a live, non-empty array")]
    let first = unsafe { read_first(bytes.as_ptr()) };

    println!("v = {v:#x}, first = {first}");

    // --- violations (uncomment one at a time) ------------------------------

    // (1) Completely unchecked call -> LINK error naming fn + missing tags:
    //   undefined reference to
    //   `SAFETY_VIOLATION__in_crate_unsafe_lib__unchecked_call_to_read__requires_tag_ValidPtr`
    //   `SAFETY_VIOLATION__in_crate_unsafe_lib__unchecked_call_to_read__requires_tag_Init`
    // let _ = unsafe { read(&x) };

    // (2) Incomplete tag accounting -> COMPILE error (name doesn't resolve):
    //   no associated item named `__safety_<hash>` found for enum `read` —
    //   the {ValidPtr} hash doesn't match the required {ValidPtr, Init} set.
    // #[safety::checked(ValidPtr = "reference is valid")]
    // let _ = unsafe { read(&x) };

    // (3) Bogus delegation -> COMPILE error: `main` has no
    //   #[safety::requires(ValidPtr = ...)], so the marker doesn't resolve:
    //   cannot find value `__safety_delegates_ValidPtr`.
    // #[safety::checked(Init = "u8 is always initialized", delegate(ValidPtr))]
    // let _ = unsafe { read(&x) };

    // (4) Missing reason string -> COMPILE error from the macro itself:
    //   safety tag `ValidPtr` is missing its reason string.
    // #[safety::checked(ValidPtr, Init = "x is initialized")]
    // let _ = unsafe { read(&x) };
}

/// Reads a value of type `T` from `ptr`.
#[safety::requires(
    ValidPtr = "`ptr` is non-null, properly aligned, and valid for reads of `T`",
    Init = "`ptr` points to a properly initialized value of type `T`"
)]
pub unsafe fn read<T>(ptr: *const T) -> T {
    core::ptr::read(ptr)
}

/// Reads the first byte of a region starting at `base`.
///
/// Discharges `Init` locally, but *delegates* `ValidPtr` upward: our own
/// caller must still prove the pointer is valid, so we re-require it.
#[safety::requires(ValidPtr = "`base` is non-null, aligned, and valid for reads of 1 byte")]
pub unsafe fn read_first(base: *const u8) -> u8 {
    // `delegate(ValidPtr)` only compiles here because this fn itself has
    // #[safety::requires(ValidPtr = ...)] — the attribute injected a
    // body-local `__safety_delegates_ValidPtr` marker. The inner
    // #[safety::checked] attribute is consumed before rustc sees it.
    #[safety::checked(
        Init = "every byte of `u8` is a valid, initialized value",
        delegate(ValidPtr)
    )]
    unsafe {
        read(base)
    }
}
