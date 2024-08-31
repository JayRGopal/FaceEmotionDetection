def check_nans_in_dict(vectors_dict):
    return {key: np.isnan(arr).any() for key, arr in vectors_dict.items()}

print(check_nans_in_dict(openface_vectors_dict))
print(check_nans_in_dict(opengraphau_vectors_dict))
print(check_nans_in_dict(hsemotion_vectors_dict))
print(check_nans_in_dict(ogauhsemotion_vectors_dict))