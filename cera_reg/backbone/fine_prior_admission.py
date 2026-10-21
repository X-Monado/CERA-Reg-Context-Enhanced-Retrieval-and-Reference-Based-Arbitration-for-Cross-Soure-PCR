"""GT-free conservative admission: a fine prior must beat the displaced one.

Fixed5% score margin is an explicitly exploratory development choice, not a
statistical uncertainty estimate. Inputs are pre-refinement descriptor/geometry
scores only; labels, dense fits and final metrics are never consulted.
"""
MARGIN=1.05


def score(c):return .5*c['geometry']+.5*c['evidence']


def admit(coarse_components,fine_components):
    if not fine_components:return False
    existing=sorted([score(c) for c in coarse_components[1:]],reverse=True)
    return len(existing)<2 or max(score(c) for c in fine_components)>MARGIN*existing[1]


def relabel_cascade(retain,candidate,accepted,success):
    # Post-decision diagnostic labels only; this function never chooses output.
    selected=candidate if accepted else retain
    labels=dict(selected['cascade'])
    labels['fine_pose_labels']=candidate['cascade']['fine_pose_labels']
    labels['sparse_viable']=any(x['success'] for x in labels['coarse_pose_labels']+labels['fine_pose_labels'])
    if success:category='success'
    elif labels['final_candidate_viable']:category='final_score_rejects_viable_pose'
    elif labels['selected_viable']:category='dense_loses_viable_prior'
    elif labels['sparse_viable']:category='prior_prunes_viable_pose'
    else:category='no_viable_sparse_pose'
    labels['category']=category
    return labels
