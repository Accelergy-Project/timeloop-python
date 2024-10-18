import cProfile
import pstats


from playground import run, get_compatibility_sets

if __name__ == "__main__":
    cs = get_compatibility_sets()
    with cProfile.Profile() as pr:
        run(cs)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
