def compute_common_prefix_of_two_strings(a, b):
    """
    计算两个字符串的最长公共前缀
    :param a:
    :param b:
    :return:
    """
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i
