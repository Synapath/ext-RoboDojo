"""Finish every render-product release, including already-removed SDG nodes."""


def release_render_product(product, annotators, path):
    warnings, errors = [], []
    for name, annotator in annotators.items():
        try:
            annotator.detach([path])
        except (TypeError, RuntimeError) as error:
            # Replicator's shared dependency can already be gone when another
            # annotator/product detached it. This is not a valid live NodeObj.
            if "Invalid NodeObj" in str(error):
                warnings.append({"annotator": name, "reason": str(error)})
            else:
                errors.append(error)
        except Exception as error:
            errors.append(error)
    try:
        product.destroy()
    except Exception as error:
        errors.append(error)
    if errors:
        raise RuntimeError("Render-product release failed after all cleanup operations") from errors[0]
    return warnings
