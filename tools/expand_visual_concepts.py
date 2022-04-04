import argparse
from .imagenet1k_oai import imagenet1k_classnames_oai

def expand_concepts(seed_file):
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    parser.add_argument('--seed_concepts', type=str, metavar="FILE", help='path to seed file', )
    args, unparsed = parser.parse_known_args()

    import pdb; pdb.set_trace()
    if args.seed_concepts:
        seed_concepts = args.seed_concepts
    else:
        seed_concepts = imagenet1k_classnames_oai

    # expand visual concepts
    concepts = expand_concepts(args.seed_concepts)