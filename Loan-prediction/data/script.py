#! /usr/bin/env python3

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
df_ch.plot(kind='bar',ax=ax1)

ax2 = fig.add_subplot(122)
df_prob.plot(kind='bar', ax=ax2)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

plt.tight_layout()
plt.savefig('LoanStatus-and-CreditHistory.png')
plt.close()
