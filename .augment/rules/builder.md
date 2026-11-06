---
type: "always_apply"
description: "Example description"
---

# ✅ Augment Rule: `builder_convention_rule`

**Description**
Enforces best practices and conventions for builder classes (`xxx_builder`) to ensure consistency, readability, and correctness within the `memory` namespace.

No such class exists in this repository yet; this rule documents the convention to follow if one is added.

---

## 🎯 Targets

- **Class names** ending with `_builder`

---

## 📏 Rule Checks

### 🔹 Class Naming
- **Must** end with `_builder`
  💬 _"Builder class name must end with `_builder`."_

---

### 🔹 Member Variables
- **Preferred**: Include `ptr_mutable<xxx>`
  💬 _"Prefer using `ptr_mutable<xxx>` to hold the object under construction."_

---

### 🔹 Methods

#### ✔ Naming
- **Must** start with `with_` followed by the variable name
  💬 _"Setter methods must follow the `with_<field>` naming convention."_

#### ✔ Parameters
- **Must** accept exactly one parameter
  💬 _"Each setter method must accept exactly one parameter."_

#### ✔ Return Type
- **Must** start with `memory::`
  💬 _"Setter methods must return a type starting with `memory::` to support fluent builder chaining."_

#### ✔ Comments
- **Must** include descriptive comments
  💬 _"Each method must have a descriptive comment explaining its purpose."_

---

### 🔹 Class-Level Comment
- **Must** include a top-level comment
  💬 _"The builder class must include a class-level comment describing its role."_

---

## ✅ Summary

| Element                         | Rule Type     | Requirement                                                   |
|----------------------------------|---------------|-----------------------------------------------------------------|
| **Class name**                  | Must          | Ends with `_builder`                                          |
| **`ptr_mutable<xxx>` member**   | Preferred     | Used to hold the object being constructed                     |
| **Setter method name**          | Must          | Follows `with_<field>` naming                                 |
| **Setter parameter count**      | Must          | Exactly one parameter                                         |
| **Setter return type**          | Must          | Starts with `memory::`                                        |
| **Method comments**             | Must          | Each method must be documented                                |
| **Class-level comment**         | Must          | Describes the role of the builder                             |
