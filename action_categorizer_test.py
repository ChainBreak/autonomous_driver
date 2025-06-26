import numpy as np
import pytest
from action_categorizer import ActionCategorizer


@pytest.mark.parametrize("action_length", [3, 4, 5])
def test_action_categorizer_roundtrip(action_length):
    """Test that ActionCategorizer can convert action vectors to categories and back correctly."""
    categorizer = ActionCategorizer(action_length)
    
    # Test all possible action vectors
    for category in range(categorizer.num_categories):
        # Convert category to action vector
        action_vector = categorizer.to_action(category)
        
        # Convert action vector back to category
        roundtrip_category = categorizer.to_category(action_vector)
        
        # Verify roundtrip conversion works
        assert roundtrip_category == category, f"Roundtrip failed for category {category} with action length {action_length}"
        
        # Verify action vector has correct shape and type
        assert action_vector.shape == (action_length,)
        
