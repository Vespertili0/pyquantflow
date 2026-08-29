import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pyquantflow.strategies.example_strategy import SmaCross


class TestSmaCross(unittest.TestCase):
    @patch("pyquantflow.strategies.example_strategy.crossover")
    @patch(
        "pyquantflow.strategies.example_strategy.SmaCross.position",
        new_callable=PropertyMock,
    )
    def test_next_buy_no_position(self, mock_position, mock_crossover):
        strategy = SmaCross.__new__(SmaCross)
        strategy.sma1 = MagicMock()
        strategy.sma2 = MagicMock()
        mock_position.return_value = False
        strategy.buy = MagicMock()
        strategy.sell = MagicMock()

        def side_effect(s1, s2):
            return s1 is strategy.sma1 and s2 is strategy.sma2

        mock_crossover.side_effect = side_effect

        strategy.next()

        strategy.buy.assert_called_once()
        strategy.sell.assert_not_called()

    @patch("pyquantflow.strategies.example_strategy.crossover")
    @patch(
        "pyquantflow.strategies.example_strategy.SmaCross.position",
        new_callable=PropertyMock,
    )
    def test_next_buy_with_position(self, mock_position, mock_crossover):
        strategy = SmaCross.__new__(SmaCross)
        strategy.sma1 = MagicMock()
        strategy.sma2 = MagicMock()
        mock_position.return_value = True
        strategy.buy = MagicMock()
        strategy.sell = MagicMock()

        def side_effect(s1, s2):
            return s1 is strategy.sma1 and s2 is strategy.sma2

        mock_crossover.side_effect = side_effect

        strategy.next()

        strategy.buy.assert_not_called()
        strategy.sell.assert_not_called()

    @patch("pyquantflow.strategies.example_strategy.crossover")
    @patch(
        "pyquantflow.strategies.example_strategy.SmaCross.position",
        new_callable=PropertyMock,
    )
    def test_next_sell_with_position(self, mock_position, mock_crossover):
        strategy = SmaCross.__new__(SmaCross)
        strategy.sma1 = MagicMock()
        strategy.sma2 = MagicMock()
        mock_position.return_value = True
        strategy.buy = MagicMock()
        strategy.sell = MagicMock()

        def side_effect(s1, s2):
            return s1 is strategy.sma2 and s2 is strategy.sma1

        mock_crossover.side_effect = side_effect

        strategy.next()

        strategy.sell.assert_called_once()
        strategy.buy.assert_not_called()

    @patch("pyquantflow.strategies.example_strategy.crossover")
    @patch(
        "pyquantflow.strategies.example_strategy.SmaCross.position",
        new_callable=PropertyMock,
    )
    def test_next_sell_no_position(self, mock_position, mock_crossover):
        strategy = SmaCross.__new__(SmaCross)
        strategy.sma1 = MagicMock()
        strategy.sma2 = MagicMock()
        mock_position.return_value = False
        strategy.buy = MagicMock()
        strategy.sell = MagicMock()

        def side_effect(s1, s2):
            return s1 is strategy.sma2 and s2 is strategy.sma1

        mock_crossover.side_effect = side_effect

        strategy.next()

        strategy.sell.assert_not_called()
        strategy.buy.assert_not_called()

    @patch("pyquantflow.strategies.example_strategy.crossover")
    @patch(
        "pyquantflow.strategies.example_strategy.SmaCross.position",
        new_callable=PropertyMock,
    )
    def test_next_no_crossover(self, mock_position, mock_crossover):
        strategy = SmaCross.__new__(SmaCross)
        strategy.sma1 = MagicMock()
        strategy.sma2 = MagicMock()
        mock_position.return_value = False
        strategy.buy = MagicMock()
        strategy.sell = MagicMock()

        mock_crossover.return_value = False

        strategy.next()

        strategy.buy.assert_not_called()
        strategy.sell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
