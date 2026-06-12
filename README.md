# Safety Tags

An implementation of the [Safety Tags RFC](https://github.com/rust-lang/rfcs/pull/3842) in stable rust, usable today!

```rust
 // 💡 define safety tags on an unsafe function
#[safety::requires(
    valid_ptr = "src must be [valid](https://doc.rust-lang.org/std/ptr/index.html#safety) for reads",
    aligned = "src must be properly aligned, even if T has size 0",
    initialized = "src must point to a properly initialized value of type T"
)]
unsafe fn read<T>(ptr: *const T) -> T {}

#[safety::checks]
fn main() {
    // 💡 you MUST discharge safety tags on an unsafe call
    #[safety::checked(
        valid_ptr = "`&()` is a reference, hence non-null, aligned, valid for reads",
        init = "references are always fully initialized"
        aligned = "`&()` is a reference, hence non-null, aligned, valid for reads"
    )]
    unsafe { read(&()) };
}
```

## Delegation

Instead of discharging the safety tag at the callsite, we can delegate it to te caller of the method. A formalization of the classic `// Safety: ensured by caller` pattern.

```rust
#[safety::requires(valid_ptr = "`base` is non-null, aligned, and valid for reads of 1 byte")]
pub unsafe fn read_first(base: *const u8) -> u8 {
    // 💡 instead of discharging at the callsite we can **delegate** the discharge requirement to our callers!
    #[safety::checked(
        init = "every byte of `u8` is a valid, initialized value",
        delegate(valid_ptr = "ensured by the caller")
    )]
    unsafe {
        read(base)
    }
}
```
