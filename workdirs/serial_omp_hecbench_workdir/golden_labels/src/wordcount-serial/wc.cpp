#include <functional>
#include <numeric>
#include <vector>















inline bool is_alpha(const char c)
{
  return (c >= 'A' && c <= 'z');
}



struct is_word_start
{
  bool operator()(const char& left, const char& right) const
  {
    return is_alpha(right) && !is_alpha(left);
  }
};

int word_count(const std::vector<char> &input)
{
  

  if (input.empty()) return 0;

  

  const char *in = input.data();
  const size_t size = input.size();
  
  

  int wc = 0;
    {
        for (int i = 0; i < size - 1; i++) {
      wc += !is_alpha(in[i]) && is_alpha(in[i+1]);
    }
  }

  

  if (is_alpha(in[0])) wc++;

  return wc;
}

int word_count_reference(const std::vector<char> &input)
{
  

  if (input.empty()) return 0;

  

  int wc = std::inner_product(
      input.cbegin(), input.cend() - 1, 

      input.cbegin() + 1,               

      0,                                

      std::plus<int>(),                 

      is_word_start());                 


  

  if (is_alpha(input.front())) wc++;

  return wc;
}